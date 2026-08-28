import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln

# ==============================================================================

# ==============================================================================
def continuous_binom(x_val, k_tensor):

    if x_val < 0:
        beta = -x_val
        sign = (-1) ** k_tensor
        val = torch.exp(gammaln(beta + k_tensor) - gammaln(k_tensor + 1) - gammaln(torch.tensor(beta, dtype=torch.float32)))
        return sign * val
    else:
        return torch.exp(gammaln(x_val + 1) - gammaln(k_tensor + 1) - gammaln(x_val - k_tensor + 1))

class FractionalImageProcessor(nn.Module):

    def __init__(self, alpha, window: int, channels: int):
        super().__init__()
        assert window % 2 == 1, "window must be odd number"
        self.pad = window // 2
        self.channels = channels
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(window) - self.pad,
            torch.arange(window) - self.pad,
            indexing='ij'
        )
        radius = torch.sqrt(grid_x ** 2 + grid_y ** 2).round()
        
        coeff = ((-1) ** radius) * continuous_binom(alpha, radius)
        coeff = coeff / (coeff.abs().sum() + 1e-10)
        weight = coeff.view(1, 1, window, window).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', weight)

    def forward(self, img):
        pad_mode = 'reflect' if min(img.shape[2:]) > self.pad else 'replicate'
        padding_img = F.pad(img, pad=(self.pad, self.pad, self.pad, self.pad), mode=pad_mode)
        kernel = self.kernel.to(device=img.device, dtype=img.dtype)
        return F.conv2d(padding_img, kernel, groups=self.channels)

class CosineLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, input, target):
        input_flat = input.flatten(1).float()
        target_flat = target.flatten(1).float()
        
        input_norm = input_flat.norm(dim=1, keepdim=True)
        target_norm = target_flat.norm(dim=1, keepdim=True)
        
        denominator = (input_norm * target_norm).clamp_min(self.eps)
        similarity = (input_flat * target_flat).sum(dim=1, keepdim=True) / denominator
        
        both_zero = (input_norm <= self.eps) & (target_norm <= self.eps)
        similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
        
        return 1.0 - similarity.squeeze(1).clamp(-1.0, 1.0)

# ==============================================================================
# 2. ºËÐÄ£ºSCA-IIT (Anchor-Directed Scale Conflict Attenuation)
# ==============================================================================
class SCA_IIT_Loss(nn.Module):
    def __init__(self, channels=3, base_lambdas=(2.0, 1.0, 0.5, 0.25), beta_cos=2.0):

        super().__init__()
        self.base_lambdas = tuple(float(l) for l in base_lambdas)
        self.num_levels = len(base_lambdas)
        
        if self.num_levels < 2:
            raise ValueError("SCA-IIT requires at least 2 levels (num_levels >= 2).")
        if not all(l > 0 for l in self.base_lambdas):
            raise ValueError("All empirical static scale coefficients (base_lambdas) must be strictly positive.")
            
        self.beta_cos = float(beta_cos)
        self.cos_loss = CosineLoss()
        self.frac_processor = FractionalImageProcessor(alpha=-1.6, window=13, channels=channels)

    def iit_response_pyramid(self, image):
        min_required_size = 2 ** (self.num_levels - 1)
        if min(image.shape[-2:]) < min_required_size:
            raise ValueError(f"Input image spatial size {image.shape[-2:]} is too small for {self.num_levels}-level pyramid.")
            
        current = image
        pyr = []
        for i in range(self.num_levels - 1):
            h, w = current.shape[2:]
            down = F.interpolate(current, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
            up = self.frac_processor(current)
            pyr.append(current - up)
            current = down
            
        pyr.append(current) 
        return pyr

    def compute_Lm0_per_sample(self, pred_lvl, gt_lvl, lambda_m):
        l1_part = lambda_m * torch.log1p(torch.abs(pred_lvl - gt_lvl)).mean(dim=(1, 2, 3))
        cos_part = self.beta_cos * self.cos_loss(pred_lvl, gt_lvl)
        return l1_part + cos_part

    def forward(self, pred, target, return_diagnostics=False):
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        if pred.shape[1] != self.frac_processor.channels:
            raise ValueError(f"Channel mismatch: expected {self.frac_processor.channels}, got {pred.shape[1]}")
            
        B = pred.size(0)
        M = self.num_levels
        anchor_idx = M - 1
        
        pred_probe = pred.detach().clone().float().requires_grad_(True)
        target_fp32 = target.detach().float()
        
        pyr_pred_probe = self.iit_response_pyramid(pred_probe)
        pyr_gt = self.iit_response_pyramid(target_fp32)
        
        g_list = []
        for m in range(M):
            L_m0_probe = self.compute_Lm0_per_sample(pyr_pred_probe[m], pyr_gt[m], self.base_lambdas[m])
            g_m = torch.autograd.grad(
                outputs=L_m0_probe.sum(), inputs=pred_probe,
                retain_graph=True, create_graph=False, only_inputs=True
            )[0]
            g_list.append(g_m.float().reshape(B, -1)) # [B, D]


        g_anchor = g_list[anchor_idx]
        v = torch.stack([(g_m * g_anchor).sum(dim=1) for g_m in g_list], dim=1) # [B, M]
        

        a = torch.full((B, M), 1.0 / M, device=pred.device)

        
        P_total = (a * v).sum(dim=1) 
        mask_neg = P_total < 0       
        
        gamma_vals = torch.ones(B, device=pred.device)
        
        if mask_neg.any():
            v_pos = torch.clamp(v, min=0.0)
            v_neg = torch.clamp(v, max=0.0)
            
            P_pos = (a * v_pos).sum(dim=1)
            P_neg = (a * v_neg).sum(dim=1)
            
            gamma = (P_pos / (-P_neg + 1e-8)).clamp_(0.0, 1.0)
            gamma_vals = torch.where(mask_neg, gamma, gamma_vals)
            
            gamma_mult = torch.where((v < 0) & mask_neg.unsqueeze(1), gamma.unsqueeze(1), torch.ones_like(a))
            a = a * gamma_mult
            
            a = a / a.sum(dim=1, keepdim=True)

        P_after = (a * v).sum(dim=1) 
        a_scaled = (a * M).detach() 
        

        pyr_pred_real = self.iit_response_pyramid(pred)
        pyr_gt_real = self.iit_response_pyramid(target)
        
        total_loss = 0.0
        for m in range(M):
            L_m0_real = self.compute_Lm0_per_sample(pyr_pred_real[m], pyr_gt_real[m], self.base_lambdas[m])
            total_loss += (a_scaled[:, m] * L_m0_real).mean()
            
        if not return_diagnostics:
            return total_loss
            
        base_lambdas_tensor = torch.tensor(self.base_lambdas, device=pred.device)
        
        diagnostics = {
            "anchor_violation_ratio": mask_neg.float().mean().item(),        
            "suppressed_scales": ((v < 0) & mask_neg.unsqueeze(1)).sum(dim=1).float().mean().item(), 
            "gamma_mean": gamma_vals.mean().item(),                          
            "anchor_proj_before": P_total.detach().mean().item(),            
            "anchor_proj_after_min": P_after.min().item(),                   
            "task_weights": a_scaled.mean(dim=0).cpu().numpy().tolist(),    
            "effective_robust_weights": (a_scaled * base_lambdas_tensor).mean(dim=0).cpu().numpy().tolist()
        }
        
        return total_loss, diagnostics