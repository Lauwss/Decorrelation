import argparse
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data import UnpairedImageDataset
from model import CUTGenerator, PatchDiscriminator, PatchSampleF
from utils.loss import GANLoss, MultiLayerPatchNCELoss
from utils.PSNR_evaluate import evaluate_paired_psnr
from utils.training import TrainingMonitor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_count(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad)
    return total, trainable


def print_training_setup(args, device, use_amp, generator, discriminator,
                         feature_net):
    print('Training configuration:')
    for name, value in vars(args).items():
        print(f'  {name}={value}')
    print(f'  resolved_device={device}')
    print(f'  amp_enabled={use_amp}')
    print('Model parameters:')
    for name, module in (('G', generator), ('D', discriminator),
                         ('F', feature_net)):
        total, trainable = parameter_count(module)
        print(f'  {name}: total={total:,} trainable={trainable:,} '
              f'({total * 4 / 1024 ** 2:.2f} MiB fp32)')


def set_epoch_learning_rate(optimizers, epoch, constant_epochs,
                            cosine_epochs, initial_lr, minimum_lr):
    if epoch <= constant_epochs:
        learning_rate = initial_lr
    else:
        cosine_epoch = epoch - constant_epochs
        cosine_factor = 0.5 * (1 + math.cos(
            math.pi * cosine_epoch / cosine_epochs))
        learning_rate = minimum_lr + (initial_lr - minimum_lr) * cosine_factor
    for optimizer in optimizers:
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = learning_rate
    return learning_rate


def train_one_step(generator, discriminator, feature_net, batch,
                   optimizer_g, optimizer_d, gan_loss, nce_loss,
                   device,lambda_g, lambda_nce, scaler, use_amp):
    real_a = batch['A'].to(device, non_blocking=True)
    real_b = batch['B'].to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        fake_b, fake_features = generator(real_a, return_features=True)

    optimizer_d.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss_d = (gan_loss(discriminator(real_b), True) +
                  gan_loss(discriminator(fake_b.detach()), False)) * 0.5
    scaler.scale(loss_d).backward()
    scaler.step(optimizer_d)

    optimizer_g.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        _, real_features = generator(real_a, return_features=True)
    selected = ['stem', 'downsample_1', 'downsample_2', 'resblock_4', 'resblock_8']
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        fake_patches, patch_ids = feature_net(
            [fake_features[name] for name in selected], num_patches=64)
        with torch.no_grad():
            real_patches, _ = feature_net(
                [real_features[name] for name in selected], num_patches=64,
                patch_ids=patch_ids)
        loss_nce = nce_loss(fake_patches, real_patches, 64)
        loss_g = lambda_g * gan_loss(discriminator(fake_b), True) + lambda_nce * loss_nce
    scaler.scale(loss_g).backward()
    scaler.step(optimizer_g)
    scaler.update()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    return {'loss_g': loss_g.item(), 'loss_d': loss_d.item(),
            'loss_nce': loss_nce.item()}


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal unpaired CUT training')
    parser.add_argument('--data-root', default=config.DATA_ROOT)
    parser.add_argument('--crop-size', type=int, default=config.CROP_SIZE)
    parser.add_argument('--load-size', type=int, default=config.LOAD_SIZE)
    parser.add_argument('--disable-weak-degradation', action='store_true',
                        default=config.DISABLE_WEAK_DEGRADATION)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--n-epochs-constant', type=int,
                        default=config.N_EPOCHS_CONSTANT)
    parser.add_argument('--n-epochs-cosine', type=int,
                        default=config.N_EPOCHS_COSINE)
    parser.add_argument('--ngf', type=int, default=config.NGF)
    parser.add_argument('--n-blocks', type=int, default=config.N_BLOCKS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--min-lr', type=float, default=config.MIN_LEARNING_RATE)
    parser.add_argument('--lambda-g', type=float, default=config.LAMBDA_G)
    parser.add_argument('--lambda-nce', type=float, default=config.LAMBDA_NCE)
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR)
    parser.add_argument('--psnr-max-images', type=int,
                        default=config.PSNR_MAX_IMAGES)
    parser.add_argument('--psnr-interval-epochs', type=int,
                        default=config.PSNR_INTERVAL_EPOCHS)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--disable-amp', action='store_true',
                        default=not config.USE_AMP)
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=config.SEED)
    args = parser.parse_args()
    if args.n_epochs_constant < 0:
        parser.error('--n-epochs-constant must be non-negative')
    if args.n_epochs_cosine < 1:
        parser.error('--n-epochs-cosine must be positive')
    if args.min_lr < 0 or args.min_lr > args.lr:
        parser.error('--min-lr must be between zero and --lr')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    if args.psnr_interval_epochs < 1:
        parser.error('--psnr-interval-epochs must be positive')
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable. GPU training must run inside a Slurm job '
            'with a CUDA-enabled PyTorch environment.')
    device = torch.device(args.device)
    use_amp = device.type == 'cuda' and not args.disable_amp
    dataset = UnpairedImageDataset(
        args.data_root,
        crop_size=args.crop_size,
        load_size=args.load_size,
        weak_degradation=not args.disable_weak_degradation,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    monitor = TrainingMonitor(args.output_dir)
    generator = CUTGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    discriminator = PatchDiscriminator(ndf=args.ngf).to(device)
    feature_net = PatchSampleF(nc=256).to(device)
    with torch.no_grad():
        _, initialization_features = generator(
            torch.zeros(1, 3, args.crop_size, args.crop_size, device=device),
            return_features=True)
    selected = ['stem', 'downsample_1', 'downsample_2', 'resblock_4', 'resblock_8']
    feature_net.initialize([initialization_features[name] for name in selected])
    optimizer_g = torch.optim.Adam(
        list(generator.parameters()) + list(feature_net.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                   betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    nce_loss = MultiLayerPatchNCELoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print_training_setup(
        args, device, use_amp, generator, discriminator, feature_net)
    total_epochs = args.n_epochs_constant + args.n_epochs_cosine
    for epoch in range(1, total_epochs + 1):
        set_epoch_learning_rate(
            (optimizer_g, optimizer_d), epoch, args.n_epochs_constant,
            args.n_epochs_cosine, args.lr, args.min_lr)
        epoch_metrics = {'loss_g': 0.0, 'loss_d': 0.0, 'loss_nce': 0.0}
        epoch_start = time.perf_counter()
        for batch in loader:
            metrics = train_one_step(
                generator, discriminator, feature_net, batch, optimizer_g,
                optimizer_d, gan_loss, nce_loss, device, args.lambda_g,  args.lambda_nce,
                scaler, use_amp)
            for name, value in metrics.items():
                epoch_metrics[name] += value
        for name in epoch_metrics:
            epoch_metrics[name] /= len(loader)
        epoch_metrics['epoch_time_seconds'] = time.perf_counter() - epoch_start
        epoch_metrics['lr_g'] = optimizer_g.param_groups[0]['lr']
        epoch_metrics['lr_d'] = optimizer_d.param_groups[0]['lr']
        if epoch % args.psnr_interval_epochs == 0:
            epoch_metrics['train_psnr'] = evaluate_paired_psnr(
                generator, f'{args.data_root}/trainA', f'{args.data_root}/trainB',
                device, args.psnr_max_images)
            epoch_metrics['test_psnr'] = evaluate_paired_psnr(
                generator, f'{args.data_root}/testA', f'{args.data_root}/testB',
                device, args.psnr_max_images)
        else:
            epoch_metrics['train_psnr'] = None
            epoch_metrics['test_psnr'] = None
        is_best = monitor.record(
            epoch, epoch_metrics, generator, discriminator, feature_net,
            optimizer_g, optimizer_d)
        psnr_message = ''
        if epoch_metrics['test_psnr'] is not None:
            psnr_message = (
                f" train_psnr={epoch_metrics['train_psnr']:.4f} "
                f"test_psnr={epoch_metrics['test_psnr']:.4f}"
            )
        status = ' best_test_psnr' if is_best else ''
        print(
            f"epoch={epoch}/{total_epochs} "
            f"loss_g={epoch_metrics['loss_g']:.4f} "
            f"loss_d={epoch_metrics['loss_d']:.4f} "
            f"loss_nce={epoch_metrics['loss_nce']:.4f} "
            f"lr_g={epoch_metrics['lr_g']:.8f} "
            f"lr_d={epoch_metrics['lr_d']:.8f} "
            f"epoch_time={epoch_metrics['epoch_time_seconds']:.2f}s"
            f"{psnr_message}{status}"
        )


if __name__ == '__main__':
    main()