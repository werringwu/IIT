import os
import torch
import yaml
import csv
import time
import numpy as np
import random
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ptflops import get_model_complexity_info

# ==================== Custom Modules ====================
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from transform.model_utils import *
from transform.dir_utils import *
from transform.image_utils import *
from transform.data_RGB import get_training_data, get_validation_data2
from transform.scheduler import GradualWarmupScheduler

# Import SCA-IIT Loss
from basicsr.models.losses.SCA_IIT_arch import SCA_IIT_Loss
# ========================================================

parser = argparse.ArgumentParser(description='Hyper-parameters for MDFormer')
parser.add_argument('--gpu', default='0', type=str, help='Specify GPU ID')
parser.add_argument('--opt', default="./training.yaml", type=str)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f"Using GPU(s): {args.gpu}")

# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

# Load yaml configuration file
yaml_file = args.opt
with open(yaml_file, 'r') as config:
    opt = yaml.safe_load(config)
print("load training yaml file: %s" % (yaml_file))

Train = opt['TRAINING']
OPT = opt['OPTIM']

# Build Model
print('==> Build the model')
device = torch.device('cuda')
model_restored = RetinexFormer(stage=1, n_feat=40, num_blocks=[1, 2, 2]).cuda()
macs, params = get_model_complexity_info(model_restored, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)

# Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Resume
if Train['RESUME']:
    path_chk_rest = get_last_path(model_dir, '_latest.pth')
    load_checkpoint(model_restored, path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    load_optim(optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print(f"==> Resuming Training with learning rate: {new_lr}")

# ==================== Initialize SCA-IIT ====================
print('==> Initializing SCA-IIT Strategy')
criterion = SCA_IIT_Loss(channels=3, base_lambdas=(2.0, 1.0, 0.5, 0.25), beta_cos=2.0).cuda()
# ============================================================

# DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True, num_workers=8, drop_last=False)

val_dataset = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

train_config = f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {params}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    Model FLOPs:        {macs}
    Strategy:           Retinexformer + SCA-IIT Multi-scale Loss
------------------------------------------------------------------'''
print(train_config)

# Logging Setup
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
mkdir(log_dir)
log_txt_path = os.path.join(log_dir, 'training_log.txt')

def write_log(content, is_header=False):
    print(content)
    with open(log_txt_path, 'a', encoding='utf-8') as f:
        if is_header:
            f.write('\n' + '=' * 80 + '\n')
        f.write(content + '\n')

total_start_time = time.time()
init_log = f'''==> Training start at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(total_start_time))}
{train_config}'''
write_log(init_log, is_header=True)

log_csv_path = os.path.join(log_dir, 'training_log.csv')
with open(log_csv_path, 'w', newline='', encoding='utf-8') as fd:
    csv_writer = csv.DictWriter(fd, fieldnames=[
        'Epoch', 'Train_Loss', 'LR', 'W_Level0', 'W_Level1', 'W_Level2', 'W_Level3',
        'Val_PSNR', 'Val_SSIM', 'Best_PSNR', 'Best_Epoch_PSNR',
        'Best_SSIM', 'Best_Epoch_SSIM', 'Epoch_Time(s)'
    ])
    csv_writer.writeheader()

best_psnr, best_ssim = 0, 0
best_epoch_psnr, best_epoch_ssim = 0, 0

# -------------------------- Training Loop --------------------------
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    last_sca_weights = [0.0, 0.0, 0.0, 0.0]

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        target = data[0].cuda()
        input_ = data[1].cuda()

        # ============================================================
        # SCA-IIT Optimization
        # ============================================================
        optimizer.zero_grad()
        
        restored = model_restored(input_)
        
        # SCA-IIT handles internal gradients and anchor attenuation, returning scalar loss and diagnostics
        loss, diagnostics = criterion(restored, target, return_diagnostics=True)
        
        loss.backward()

        # Optional: Clip gradients explicitly if exploding gradients occur
        # torch.nn.utils.clip_grad_norm_(model_restored.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Extract dynamic weights computed by the conflict attenuation process
        last_sca_weights = diagnostics["task_weights"]

    avg_loss = epoch_loss / len(train_loader)
    current_lr = scheduler.get_lr()[0]
    epoch_time = time.time() - epoch_start_time

    # Validation Phase
    val_psnr, val_ssim = '-', '-'
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb, ssim_val_rgb = [], []
        with torch.no_grad():
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                h, w = target.shape[2], target.shape[3]

                restored = model_restored(input_)
                restored = restored[:, :, :h, :w]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(torchPSNR(res, tar))
                    ssim_val_rgb.append(torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))

        val_psnr = torch.stack(psnr_val_rgb).mean().item()
        val_ssim = torch.stack(ssim_val_rgb).mean().item()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(model_dir, "model_bestPSNR.pth"))

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(model_dir, "model_bestSSIM.pth"))

    # Logging Phase
    w_0, w_1, w_2, w_3 = last_sca_weights

    # Type check to avoid format string errors when validation is skipped
    val_psnr_str = f"{val_psnr:.4f}" if isinstance(val_psnr, (float, int)) else str(val_psnr)
    val_ssim_str = f"{val_ssim:.4f}" if isinstance(val_ssim, (float, int)) else str(val_ssim)

    write_log(f'''------------------------------------------------------------------
        Epoch: {epoch:3d} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}
        SCA Level Weights: L0(Fine): {w_0:.3f}, L1: {w_1:.3f}, L2: {w_2:.3f}, L3(Coarse): {w_3:.3f}
        Val PSNR: {val_psnr_str} | Best PSNR: {best_psnr:.4f} (Epoch {best_epoch_psnr})
        Val SSIM: {val_ssim_str} | Best SSIM: {best_ssim:.4f} (Epoch {best_epoch_ssim})
    ------------------------------------------------------------------\n''')

    with open(log_csv_path, 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.DictWriter(f, fieldnames=[
            'Epoch', 'Train_Loss', 'LR', 'W_Level0', 'W_Level1', 'W_Level2', 'W_Level3',
            'Val_PSNR', 'Val_SSIM', 'Best_PSNR', 'Best_Epoch_PSNR',
            'Best_SSIM', 'Best_Epoch_SSIM', 'Epoch_Time(s)'
        ])
        csv_writer.writerow({
            'Epoch': epoch,
            'Train_Loss': f'{avg_loss:.6f}',
            'LR': f'{current_lr:.6f}',
            'W_Level0': f'{w_0:.4f}',
            'W_Level1': f'{w_1:.4f}',
            'W_Level2': f'{w_2:.4f}',
            'W_Level3': f'{w_3:.4f}',
            'Val_PSNR': val_psnr_str,
            'Val_SSIM': val_ssim_str,
            'Best_PSNR': f'{best_psnr:.4f}',
            'Best_Epoch_PSNR': best_epoch_psnr,
            'Best_SSIM': f'{best_ssim:.4f}',
            'Best_Epoch_SSIM': best_epoch_ssim,
            'Epoch_Time(s)': f'{epoch_time:.2f}'
        })

    scheduler.step()
    torch.save({
        'epoch': epoch,
        'state_dict': model_restored.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

total_time = (time.time() - total_start_time) / 3600
final_log = f'''==> Training finished at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
Total training time: {total_time:.2f} hours
Best Val PSNR: {best_psnr:.4f} (Epoch {best_epoch_psnr})
Best Val SSIM: {best_ssim:.4f} (Epoch {best_epoch_ssim})
Log files saved to: {log_dir}'''
write_log(final_log, is_header=True)