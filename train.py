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
from model.backbone import initialize_weights
from utils.loss import GANLoss, ExpanderDecorrelationLoss, MultiLayerPatchNCELoss
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
                         feature_net, log=print):
    log('Training configuration:')
    for name, value in vars(args).items():
        log(f'  {name}={value}')
    log(f'  resolved_device={device}')
    log(f'  amp_enabled={use_amp}')
    log('Model parameters:')
    for name, module in (('G', generator), ('D', discriminator),
                         ('F', feature_net)):
        total, trainable = parameter_count(module)
        log(f'  {name}: total={total:,} trainable={trainable:,} '
            f'({total * 4 / 1024 ** 2:.2f} MiB fp32)')


def get_nce_feature_names(encoder_blocks):
    if encoder_blocks < 2:
        raise ValueError('encoder_blocks must be at least 2 for distinct PatchNCE features')
    return ['stem', 'downsample_1', 'downsample_2', 'enc_resblock_1',
            f'enc_resblock_{encoder_blocks}']


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


def expander_decorr_weight(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def train_one_step(generator, discriminator, feature_net, batch,
                   optimizer_g, optimizer_d, optimizer_f, gan_loss, nce_loss,
                   device, lambda_g, lambda_nce, num_patches, use_idt_nce,
                   idt_nce_weight, decorr_loss, decorr_branches,
                   decorr_indices, decorr_cov_weight, decorr_var_weight,
                   scaler, use_amp):
    real_a = batch['A'].to(device, non_blocking=True)
    real_b = batch['B'].to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        compute_idt_nce = idt_nce_weight > 0
        if compute_idt_nce:
            generated, combined_features = generator(
                torch.cat((real_a, real_b), dim=0), return_features=True)
            batch_size = real_a.shape[0]
            fake_b, idt_b = generated.split(batch_size, dim=0)
            real_features = {
                name: feature[:batch_size]
                for name, feature in combined_features.items()
            }
            real_features_b = {
                name: feature[batch_size:]
                for name, feature in combined_features.items()
            }
        else:
            fake_b, real_features = generator(real_a, return_features=True)
            idt_b = None
            real_features_b = None

    optimizer_d.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss_d = (gan_loss(discriminator(real_b), True) +
                  gan_loss(discriminator(fake_b.detach()), False)) * 0.5
    scaler.scale(loss_d).backward()
    scaler.step(optimizer_d)

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_f.zero_grad(set_to_none=True)
    selected = get_nce_feature_names(len(generator.encoder.resblocks))
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        _, fake_features = generator.encode(fake_b, return_features=True)
        fake_patches, patch_ids, fake_projected = feature_net(
            [fake_features[name] for name in selected],
            num_patches=num_patches, return_projected=True)
        real_patches, _, real_projected = feature_net(
            [real_features[name] for name in selected], num_patches=num_patches,
            patch_ids=patch_ids, return_projected=True)
        loss_nce = nce_loss(fake_patches, real_patches, num_patches)
        selected_fake_projected = [fake_projected[index]
                       for index in decorr_indices]
        selected_real_projected = [real_projected[index]
                       for index in decorr_indices]
        decorr_fake = selected_fake_projected if 'fake' in decorr_branches else []
        decorr_real = selected_real_projected if 'real_a' in decorr_branches else []
        if decorr_fake and decorr_real:
            loss_expander_cov, loss_expander_var, decorr_stats = decorr_loss(
                decorr_fake, decorr_real)
        elif decorr_fake:
            loss_expander_cov, loss_expander_var, decorr_stats = decorr_loss(
                decorr_fake, decorr_fake)
        elif decorr_real:
            loss_expander_cov, loss_expander_var, decorr_stats = decorr_loss(
                decorr_real, decorr_real)
        else:
            zero = loss_nce * 0.0
            loss_expander_cov = zero
            loss_expander_var = zero
            decorr_stats = {name: zero for name in (
                'mean_abs_corr', 'offdiag_energy', 'mean_std', 'min_std',
                'effective_rank', 'pc1_ratio')}
        loss_idt_nce = torch.zeros_like(loss_nce)
        if compute_idt_nce:
            assert idt_b is not None
            assert real_features_b is not None
            _, idt_features = generator.encode(idt_b, return_features=True)
            idt_patches, idt_patch_ids = feature_net(
                [idt_features[name] for name in selected],
                num_patches=num_patches)
            real_b_patches, _ = feature_net(
                [real_features_b[name] for name in selected],
                num_patches=num_patches, patch_ids=idt_patch_ids)
            loss_idt_nce = nce_loss(idt_patches, real_b_patches, num_patches)
        loss_g_gan = lambda_g * gan_loss(discriminator(fake_b), True)
        if use_idt_nce:
            loss_g_nce = 0.5 * lambda_nce * (
                loss_nce + idt_nce_weight * loss_idt_nce)
        else:
            loss_g_nce = lambda_nce * loss_nce
        loss_expander = (decorr_cov_weight * loss_expander_cov +
                 decorr_var_weight * loss_expander_var)
        loss_g = loss_g_gan + loss_g_nce + loss_expander
    scaler.scale(loss_g).backward()
    scaler.step(optimizer_g)
    scaler.step(optimizer_f)
    scaler.update()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    return {'loss_g': loss_g.item(), 'loss_g_gan': loss_g_gan.item(),
        'loss_d': loss_d.item(),
        'loss_nce': loss_nce.item(),
        'loss_idt_nce': (idt_nce_weight * loss_idt_nce).item(),
        'loss_expander_cov': loss_expander_cov.item(),
        'loss_expander_var': loss_expander_var.item(),
        'expander_mean_abs_corr': decorr_stats['mean_abs_corr'].item(),
        'expander_offdiag_energy': decorr_stats['offdiag_energy'].item(),
        'expander_mean_std': decorr_stats['mean_std'].item(),
        'expander_min_std': decorr_stats['min_std'].item(),
        'expander_effective_rank': decorr_stats['effective_rank'].item(),
        'expander_pc1_ratio': decorr_stats['pc1_ratio'].item(),
        'expander_decorr_weight': decorr_cov_weight}


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
    parser.add_argument('--encoder-blocks', type=int,
                        default=config.ENCODER_BLOCKS)
    parser.add_argument('--decoder-blocks', type=int,
                        default=config.DECODER_BLOCKS)
    parser.add_argument('--downsample-mode', choices=('stride', 'avgpool', 'antialias'),
                        default=config.DOWN_SAMPLE_MODE)
    parser.add_argument('--upsample-mode', choices=('nearest', 'bilinear', 'antialias', 'transposed'),
                        default=config.UP_SAMPLE_MODE)
    parser.add_argument('--init-type', choices=('normal', 'xavier', 'kaiming', 'orthogonal'),
                        default=config.INIT_TYPE)
    parser.add_argument('--init-gain', type=float, default=config.INIT_GAIN)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--min-lr', type=float, default=config.MIN_LEARNING_RATE)
    parser.add_argument('--lambda-g', type=float, default=config.LAMBDA_G)
    parser.add_argument('--lambda-nce', type=float, default=config.LAMBDA_NCE)
    parser.add_argument('--disable-idt-nce', action='store_true',
                        default=not config.USE_IDT_NCE)
    parser.add_argument('--idt-nce-interval', type=int,
                        default=config.IDT_NCE_INTERVAL)
    parser.add_argument('--num-patches', type=int, default=config.NUM_PATCHES)
    parser.add_argument('--use-expander-decorr', action='store_true',
                        default=config.USE_EXPANDER_DECORR)
    parser.add_argument('--disable-expander-decorr', action='store_false',
                        dest='use_expander_decorr')
    parser.add_argument('--lambda-expander-decorr', type=float,
                        default=config.LAMBDA_EXPANDER_DECORR)
    parser.add_argument('--lambda-expander-decorr-var', type=float,
                        default=config.LAMBDA_EXPANDER_DECORR_VAR)
    parser.add_argument('--expander-decorr-layers',
                        default=config.EXPANDER_DECORR_LAYERS)
    parser.add_argument('--expander-decorr-branches',
                        default=config.EXPANDER_DECORR_BRANCHES)
    parser.add_argument('--expander-decorr-min-std', type=float,
                        default=config.EXPANDER_DECORR_MIN_STD)
    parser.add_argument('--expander-decorr-warmup-epochs', type=int,
                        default=config.EXPANDER_DECORR_WARMUP_EPOCHS)
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
    if args.encoder_blocks < 2:
        parser.error('--encoder-blocks must be at least 2 for PatchNCE')
    if args.decoder_blocks < 0:
        parser.error('--decoder-blocks must be non-negative')
    if args.min_lr < 0 or args.min_lr > args.lr:
        parser.error('--min-lr must be between zero and --lr')
    if args.num_patches < 1:
        parser.error('--num-patches must be positive')
    if args.idt_nce_interval < 1:
        parser.error('--idt-nce-interval must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    if args.psnr_interval_epochs < 1:
        parser.error('--psnr-interval-epochs must be positive')
    if args.lambda_expander_decorr < 0 or args.lambda_expander_decorr_var < 0:
        parser.error('Expander decorrelation weights must be non-negative')
    if args.expander_decorr_min_std < 0:
        parser.error('--expander-decorr-min-std must be non-negative')
    if args.expander_decorr_warmup_epochs < 0:
        parser.error('--expander-decorr-warmup-epochs must be non-negative')
    branches = {branch.strip().lower() for branch in args.expander_decorr_branches.split(',')
                if branch.strip()}
    if not branches.issubset({'fake', 'real_a'}):
        parser.error('--expander-decorr-branches must contain fake and/or real_a')
    if args.use_expander_decorr and not branches:
        parser.error('at least one decorrelation branch is required when enabled')
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
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
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    monitor = TrainingMonitor(args.output_dir)
    generator = CUTGenerator(
        ngf=args.ngf,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        downsample_mode=args.downsample_mode,
        upsample_mode=args.upsample_mode,
    ).to(device)
    discriminator = PatchDiscriminator(
        ndf=args.ngf, downsample_mode=args.downsample_mode).to(device)
    feature_net = PatchSampleF(nc=256).to(device)
    initialize_weights(generator, args.init_type, args.init_gain)
    initialize_weights(discriminator, args.init_type, args.init_gain)
    with torch.no_grad():
        _, initialization_features = generator(
            torch.zeros(1, 3, args.crop_size, args.crop_size, device=device),
            return_features=True)
    selected = get_nce_feature_names(args.encoder_blocks)
    decorr_layers = [name.strip() for name in args.expander_decorr_layers.split(',')
                     if name.strip()]
    invalid_layers = set(decorr_layers) - set(selected)
    if invalid_layers:
        raise ValueError(
            f'Unknown Expander decorrelation layers: {sorted(invalid_layers)}; '
            f'available layers: {selected}')
    if args.use_expander_decorr and not decorr_layers:
        raise ValueError('at least one Expander decorrelation layer is required')
    decorr_branches = {branch.strip().lower() for branch in
                       args.expander_decorr_branches.split(',') if branch.strip()}
    decorr_indices = [selected.index(name) for name in decorr_layers]
    feature_net.initialize([initialization_features[name] for name in selected],
                           args.init_type, args.init_gain)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                   betas=(0.5, 0.999))
    optimizer_f = torch.optim.Adam(feature_net.parameters(), lr=args.lr,
                                   betas=(0.5, 0.999))
    gan_loss = GANLoss().to(device)
    nce_loss = MultiLayerPatchNCELoss().to(device)
    decorr_loss = ExpanderDecorrelationLoss(args.expander_decorr_min_std).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    monitor.log(f'Training started; CSV metrics: {monitor.log_path}')
    print_training_setup(
        args, device, use_amp, generator, discriminator, feature_net,
        log=monitor.log)
    total_epochs = args.n_epochs_constant + args.n_epochs_cosine
    for epoch in range(1, total_epochs + 1):
        set_epoch_learning_rate(
            (optimizer_g, optimizer_d, optimizer_f), epoch, args.n_epochs_constant,
            args.n_epochs_cosine, args.lr, args.min_lr)
        epoch_metrics = {'loss_g': 0.0, 'loss_g_gan': 0.0, 'loss_d': 0.0,
                         'loss_nce': 0.0, 'loss_idt_nce': 0.0,
                         'loss_expander_cov': 0.0, 'loss_expander_var': 0.0,
                         'expander_mean_abs_corr': 0.0,
                         'expander_offdiag_energy': 0.0,
                         'expander_mean_std': 0.0, 'expander_min_std': 0.0,
                         'expander_effective_rank': 0.0,
                         'expander_pc1_ratio': 0.0,
                         'expander_decorr_weight': 0.0}
        decorr_scale = (expander_decorr_weight(
            epoch, args.expander_decorr_warmup_epochs)
            if args.use_expander_decorr else 0.0)
        epoch_start = time.perf_counter()
        for step_index, batch in enumerate(loader, start=1):
            compute_idt_nce = (
                not args.disable_idt_nce and
                (step_index - 1) % args.idt_nce_interval == 0
            )
            metrics = train_one_step(
                generator, discriminator, feature_net, batch, optimizer_g,
                optimizer_d, optimizer_f, gan_loss, nce_loss, device,
                args.lambda_g, args.lambda_nce, args.num_patches,
                not args.disable_idt_nce,
                args.idt_nce_interval if compute_idt_nce else 0,
                decorr_loss, decorr_branches if args.use_expander_decorr else set(),
                decorr_indices,
                decorr_scale * args.lambda_expander_decorr,
                decorr_scale * args.lambda_expander_decorr_var,
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
                device, args.psnr_max_images, args.batch_size)
            epoch_metrics['test_psnr'] = evaluate_paired_psnr(
                generator, f'{args.data_root}/testA', f'{args.data_root}/testB',
                device, args.psnr_max_images, args.batch_size)
        else:
            epoch_metrics['train_psnr'] = None
            epoch_metrics['test_psnr'] = None
        is_best = monitor.record(
            epoch, epoch_metrics, generator, discriminator, feature_net,
            optimizer_g, optimizer_d, optimizer_f)
        psnr_message = ''
        if epoch_metrics['test_psnr'] is not None:
            psnr_message = (
                f" train_psnr={epoch_metrics['train_psnr']:.4f} "
                f"test_psnr={epoch_metrics['test_psnr']:.4f}"
            )
        status = ' best_test_psnr' if is_best else ''
        monitor.log(
            f"epoch={epoch}/{total_epochs} "
            f"loss_g={epoch_metrics['loss_g']:.4f} "
            f"loss_g_gan={epoch_metrics['loss_g_gan']:.4f} "
            f"loss_d={epoch_metrics['loss_d']:.4f} "
            f"loss_nce={epoch_metrics['loss_nce']:.4f} "
            f"loss_idt_nce={epoch_metrics['loss_idt_nce']:.4f} "
            f"loss_expander_cov={epoch_metrics['loss_expander_cov']:.6f} "
            f"loss_expander_var={epoch_metrics['loss_expander_var']:.6f} "
            f"expander_corr={epoch_metrics['expander_mean_abs_corr']:.6f} "
            f"lr_g={epoch_metrics['lr_g']:.8f} "
            f"lr_d={epoch_metrics['lr_d']:.8f} "
            f"epoch_time={epoch_metrics['epoch_time_seconds']:.2f}s"
            f"{psnr_message}{status}"
        )


if __name__ == '__main__':
    main()