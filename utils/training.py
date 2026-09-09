import csv
from pathlib import Path

import torch


class TrainingMonitor:
    """Persist training metrics and the latest and best-PSNR checkpoints."""

    fieldnames = ('epoch', 'loss_g', 'loss_d', 'loss_nce', 'lr_g', 'lr_d',
                  'epoch_time_seconds', 'train_psnr', 'test_psnr')

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / 'train_log.csv'
        self.latest_path = self.output_dir / 'latest.pt'
        self.best_path = self.output_dir / 'best_test_psnr.pt'
        self.best_test_psnr = float('-inf')
        self._initialize_log()

    def _initialize_log(self):
        if self.log_path.exists():
            with self.log_path.open(newline='', encoding='utf-8') as log_file:
                reader = csv.DictReader(log_file)
                if reader.fieldnames == list(self.fieldnames):
                    return
                rows = list(reader)
        else:
            rows = []
        with self.log_path.open('w', newline='', encoding='utf-8') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name) for name in self.fieldnames})

    def _checkpoint(self, epoch, generator, discriminator, feature_net,
                    optimizer_g, optimizer_d, metrics):
        return {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'feature_net': feature_net.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'learning_rate_g': optimizer_g.param_groups[0]['lr'],
            'learning_rate_d': optimizer_d.param_groups[0]['lr'],
            'metrics': metrics,
        }

    def record(self, epoch, metrics, generator, discriminator, feature_net,
               optimizer_g, optimizer_d):
        row = {'epoch': epoch, **metrics}
        with self.log_path.open('a', newline='', encoding='utf-8') as log_file:
            csv.DictWriter(log_file, fieldnames=self.fieldnames).writerow(row)
        checkpoint = self._checkpoint(
            epoch, generator, discriminator, feature_net, optimizer_g,
            optimizer_d, metrics)
        torch.save(checkpoint, self.latest_path)
        test_psnr = metrics.get('test_psnr')
        if test_psnr is not None and test_psnr > self.best_test_psnr:
            self.best_test_psnr = metrics['test_psnr']
            torch.save(checkpoint, self.best_path)
            return True
        return False