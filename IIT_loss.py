from torch.special import gammaln  # PyTorch 2.0+
import torch

def continuous_binom(x, k):
    return torch.exp(gammaln(x + 1) - gammaln(k + 1) - gammaln(x - k + 1))
    
def gen_isotropic_frac_kernel_pytorch(alpha,window:int, device=torch.device('cpu')):
    assert window % 2 ==1, "window must be odd number"
    center = window // 2
    if isinstance(alpha,(int,float)):
        alpha = torch.tensor(alpha,dtype=torch.float32,device=device)
    # if alpha.dim() == 0:
    #     alpha = alpha.view(1,1,1,1)
    while alpha.dim() < 4:
        alpha = alpha.unsqueeze(0)
    B, C, W, H = alpha.shape
    k = torch.arange(0, window, device=alpha.device if isinstance(alpha, torch.Tensor) else 'cpu', dtype=torch.float32)
    # 创建二维坐标网格
    grid_y,grid_x = torch.meshgrid(
        torch.arange(window,device=device) - center,
        torch.arange(window,device=device) - center,
        indexing='ij'
    )
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = torch.round(radius).long()

    radius_matrix =radius.expand(B,C,W,H,window,window)
    order = alpha.view(B,C,W,H,1,1).to(device)
    order = order.repeat(1,1,1,1,window,window)
    # coeff = ((-1) ** radius_matrix) * pytorch_binom(order,radius_matrix)
    coeff = ((-1 ** radius_matrix.float()) * continuous_binom(order,radius.float()))
    coeff = coeff/ (coeff.abs().sum(dim=(-2,-1),keepdim=True) + 1e-10)
    return coeff
def fraction_image_process(img,alpha,window:int, device=torch.device('cpu')):
    half_window = window // 2
    if isinstance(alpha,(int,float)):
        alpha = torch.tensor(alpha,dtype=torch.float32,device=device)
    if alpha.dim() < 4:
        alpha = alpha.unsqueeze(0)
    kernel = gen_isotropic_frac_kernel_pytorch(alpha,window,device)
    padding_img = F.pad(img.to(device),pad=(half_window,half_window,half_window,half_window),mode='reflect')
    unfolded_img = padding_img.unfold(2,window,1).unfold(3,window,1)
    frac_img = (unfolded_img * kernel).sum(dim=(-1,-2))
    return torch.abs(frac_img)
class CosineLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        # Flatten to [B, -1]
        input_flat = input.view(input.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Normalize
        input_norm = F.normalize(input, dim=-1, eps=self.eps)
        target_norm = F.normalize(target, dim=-1, eps=self.eps)

        # Cosine similarity
        cos_sim = (input_norm * target_norm).sum(dim=-1)
        cos_loss = 1.0 - cos_sim  # Cosine distance

        if self.reduction == 'mean':
            return cos_loss.mean()
        elif self.reduction == 'sum':
            return cos_loss.sum()
        else:
            return cos_loss  # [B]
	
class IIT_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(IIT_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cos_loss = CosineLoss()
    def lap_pyramid(self,image, n_levels=3,alpha=0.5,window=17):
        current = image  # .permute(0,3,1,2)
        pyr = []
        for i in range(n_levels):
            down = F.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode='bicubic',
                                 align_corners=True)
            # up = F.interpolate(down, size=(current.shape[2], current.shape[3]), mode='bicubic', align_corners=True)
            up = fraction_image_process(current,alpha,window,device=image.device)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr
    def frac_loss(self,pred,target,lambda_value=(1,2,4,8),alpha=-1.5,window=13,lambda_cos=0.25):
        pyr_pred = self.lap_pyramid(pred, n_levels=3,alpha=alpha,window=window)
        pyr_gt = self.lap_pyramid(target, n_levels=3,alpha=alpha,window=window)
        loss_val = 0.0  # torch.tensor([0],dtype=torch.float32).to(pred[0].device)
        assert len(pyr_pred) == len(pyr_gt)
        for i in range(len(pyr_pred)):
            pred_level = pyr_pred[i]
            gt_level = pyr_gt[i]
            assert pred_level.shape == gt_level.shape, f"Layer {i} shape mismatch"
            loss_val += lambda_value[i] * torch.mean(torch.log1p(torch.abs(pred_level - gt_level)), dim=(2, 3)) + self.cos_loss(pred_level,gt_level) *lambda_cos
        return torch.mean(loss_val)
    def forward(self, pred, target, weight=None, **kwargs):
        # return self.loss_weight * self.Dfrft_loss(pred, target) + self.loss_weight * self.Dfrft_loss_minus(pred, target)
        return self.loss_weight * self.frac_loss(pred,target,lambda_value=(2,1,0.5,0.25),alpha=-1.6,window=13)