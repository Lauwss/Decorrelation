import csv
from datetime import datetime
from pathlib import Path

import torch


class TrainingMonitor:
    """Persist training metrics and the latest and best-PSNR checkpoints."""

    fieldnames = ('epoch', 'loss_g', 'loss_g_gan', 'loss_d', 'loss_nce',
                  'loss_idt_nce', 'loss_expander_cov', 'loss_expander_var',
                  'expander_mean_abs_corr', 'expander_offdiag_energy',
                  'expander_mean_std', 'expander_min_std',
                  'expander_effective_rank', 'expander_pc1_ratio',
                  'expander_decorr_weight', 'lr_g', 'lr_d',
                  'epoch_time_seconds', 'train_psnr', 'test_psnr')

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / 'train_log.csv'
        self.text_log_path = self.output_dir / 'train.log'
        self.latest_path = self.output_dir / 'latest.pt'
        self.best_path = self.output_dir / 'best_test_psnr.pt'
        self.best_test_psnr = float('-inf')
        self._initialize_log()
        self.text_log_path.write_text('', encoding='utf-8')

    def log(self, message):
        timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
        line = f'[{timestamp}] {message}'
        print(line, flush=True)
        with self.text_log_path.open('a', encoding='utf-8') as log_file:
            log_file.write(line + '\n')
            log_file.flush()

    def _initialize_log(self):
        with self.log_path.open('w', newline='', encoding='utf-8') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def _checkpoint(self, epoch, generator, discriminator, feature_net,
                    optimizer_g, optimizer_d, optimizer_f, metrics):
        return {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'feature_net': feature_net.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_f': optimizer_f.state_dict(),
            'learning_rate_g': optimizer_g.param_groups[0]['lr'],
            'learning_rate_d': optimizer_d.param_groups[0]['lr'],
            'metrics': metrics,
        }

    def record(self, epoch, metrics, generator, discriminator, feature_net,
               optimizer_g, optimizer_d, optimizer_f):
        row = {'epoch': epoch, **metrics}
        with self.log_path.open('a', newline='', encoding='utf-8') as log_file:
            csv.DictWriter(log_file, fieldnames=self.fieldnames).writerow(row)
        checkpoint = self._checkpoint(
            epoch, generator, discriminator, feature_net, optimizer_g,
            optimizer_d, optimizer_f, metrics)
        torch.save(checkpoint, self.latest_path)
        test_psnr = metrics.get('test_psnr')
        if test_psnr is not None and test_psnr > self.best_test_psnr:
            self.best_test_psnr = metrics['test_psnr']
            torch.save(checkpoint, self.best_path)
            return True
        return False