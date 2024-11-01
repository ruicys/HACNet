
import argparse
import logging
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
# from utils.CR import ContrastLoss
from utils.CR_res import ContrastLoss_res

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='HACNet-l', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='NH-HAZE', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

parser.add_argument('--datasets_dir', default='./data', type=str, help='path to datasets dir')
parser.add_argument('--train_dataset', default='OTS', type=str, help='train dataset name')
parser.add_argument('--valid_dataset', default='SOTS', type=str, help='valid dataset name')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu




# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    total_train_images = 0  # 计算训练图片的总数

    for batch in tqdm(train_loader, desc="Training", leave=False):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        total_train_images += source_img.size(0)  # 统计每个批次的图片数

        with autocast(args.no_autocast):
            output = network(source_img)
            loss = criterion[0](output, target_img) + criterion[1](output, target_img, source_img) * 0.1

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    logging.info(f"Train Loss: {losses.avg:.4f}")
    logging.info(f"Total training images processed: {total_train_images}")
    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    model_name = args.model.replace('-', '_')
    print(f"Trying to load model: {model_name}")
    network.eval()

    total_valid_images = 0  # 计算验证图片的总数

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        total_valid_images += source_img.size(0)  # 统计每个批次的图片数

        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    logging.info(f"Validation PSNR: {PSNR.avg:.2f}")
    logging.info(f"Total validation images processed: {total_valid_images}")
    return PSNR.avg



def load_checkpoint_if_available(model, optimizer, scheduler, scaler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        logging.info(f"从 {checkpoint_path} 加载检查点")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        logging.info(f"加载成功，继续从第 {start_epoch} 轮开始训练，最佳 PSNR 为: {best_psnr:.2f}")
        return start_epoch, best_psnr
    else:
        logging.info("未找到检查点文件，从头开始训练")
        return 0, 0

if __name__ == '__main__':
    logging.info('=> 开始训练')

    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # 初始化模型
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    criterion = [nn.L1Loss(), ContrastLoss_res(ablation=False).cuda()]
    optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr']) if setting['optimizer'] == 'adam' \
        else torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    # 检查是否有之前的检查点，并加载
    checkpoint_path = os.path.join(args.save_dir, args.exp, args.model + '.pth')
    start_epoch, best_psnr = load_checkpoint_if_available(network, optimizer, scheduler, scaler, checkpoint_path)

    # 创建数据加载器
    train_dataset = PairLoader(args.datasets_dir, args.train_dataset, 'train',
                               setting['patch_size'], setting['only_h_flip'])
    valid_path = f"{args.valid_dataset}/{args.exp}"
    # val_dataset = PairLoader(args.datasets_dir, valid_path, 'valid', setting['valid_mode'], setting['patch_size'])
    val_dataset = PairLoader(args.datasets_dir, valid_path, 'valid', size=setting['patch_size'])

    train_loader = DataLoader(train_dataset, batch_size=setting['batch_size'], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    # 训练循环
    for epoch in range(start_epoch, setting['epochs'] + 1):
        logging.info(f'开始第 [{epoch}/{setting["epochs"]}] 轮训练')

        start_time = time.time()

        # 训练
        train_loss = train(train_loader, network, criterion, optimizer, scaler)
        scheduler.step()

        # 验证
        if epoch % setting['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network)
            logging.info(f'第 [{epoch}/{setting["epochs"]}] 轮 PSNR: {avg_psnr:.2f}')

            # 保存最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                logging.info(f'新的最佳 PSNR: {best_psnr:.2f}, 正在保存模型...')
                torch.save({
                    'state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'best_psnr': best_psnr
                }, checkpoint_path)

        logging.info(f'第 [{epoch}] 轮训练完成，耗时: {time.time() - start_time:.2f}s')

    logging.info(f'=> 训练完成，最佳 PSNR: {best_psnr:.2f}')