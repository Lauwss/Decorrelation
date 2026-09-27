# Expander Decorrelation 200 Epoch Report

## Scope

This report summarizes the completed 200 epoch runs on RRSHID. All Exp1 and
Exp4 runs use `lambda_nce=1.0`, `idt_nce_interval=4`, 100 constant-learning-rate
epochs, 100 cosine-decay epochs, a 20 epoch decorrelation warmup, and seeds
1407, 2407, and 3407.

- Exp1: `lambda_expander_decorr=0.0001`,
  `lambda_expander_decorr_var=0.0025`.
- Exp4: `lambda_expander_decorr=0.1`,
  `lambda_expander_decorr_var=0.25`.
- Final test: Exp4 weights, seed 1407, but `idt_nce_interval=1`.

The best PSNR is the maximum logged test PSNR. Test PSNR is evaluated every
five epochs, so its selected epoch is one of those checkpoints. Mean and
standard deviation below are calculated across the three independent seeds.

## Results

| Run | Best test PSNR (dB) | Best epoch | Final test PSNR (dB) | Final mean abs. correlation | Final effective rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp1 seed 1407 | 20.7954 | 195 | 20.7782 | 0.1025 | 54.8 |
| Exp1 seed 2407 | 21.0226 | 180 | 21.0072 | 0.1377 | 56.8 |
| Exp1 seed 3407 | 20.8530 | 155 | 20.6161 | 0.1108 | 52.6 |
| **Exp1 mean +/- std** | **20.8903 +/- 0.1181** | - | **20.8005 +/- 0.1965** | **0.1170 +/- 0.0184** | **54.7 +/- 2.1** |
| Exp4 seed 1407 | 21.3189 | 180 | 21.2275 | 0.1024 | 54.2 |
| Exp4 seed 2407 | 21.0691 | 180 | 21.0232 | 0.1045 | 52.5 |
| Exp4 seed 3407 | 21.2080 | 185 | 21.1663 | 0.1038 | 53.7 |
| **Exp4 mean +/- std** | **21.1987 +/- 0.1251** | - | **21.1390 +/- 0.1048** | **0.1036 +/- 0.0011** | **53.5 +/- 0.8** |
| Final test, Exp4, interval 1, seed 1407 | 20.6860 | 170 | 20.4518 | 0.1062 | 50.5 |

## Analysis

1. **Exp4 is the preferred decorrelation setting.** Its mean best PSNR exceeds
   Exp1 by 0.3084 dB, and its mean final PSNR exceeds Exp1 by 0.3385 dB. Every
   paired seed also favors Exp4 at the final checkpoint: +0.4493 dB (1407),
   +0.0160 dB (2407), and +0.5502 dB (3407).

2. **The stronger weights actually decorrelate features more reliably.** Exp4
   ends at 0.1036 mean absolute correlation versus 0.1170 for Exp1. More
   importantly, its standard deviation is 0.0011 rather than 0.0184, so the
   behavior is much less seed-sensitive. Effective ranks remain close (53.5
   versus 54.7), and Exp4's PC1 ratio is approximately 0.03 for every seed;
   this indicates decorrelation without a low-rank collapse.

3. **The best model is Exp4 seed 1407 at epoch 180.** It reaches 21.3189 dB and
   retains 21.2275 dB at epoch 200. Use
   `results/ckpt/expander_decorr_200_exp4_s1407/best_test_psnr.pt` for the
   currently strongest checkpoint.

4. **`idt_nce_interval=1` is not supported by this matched comparison.** With
   the same Exp4 weights and seed 1407, its best PSNR is 20.6860 dB, 0.6329 dB
   below Exp4 interval 4; its final PSNR is 0.7757 dB lower. Its effective rank
   also falls from 54.2 to 50.5. Keep `idt_nce_interval=4` for this setup.

5. **Do not use `confirm` as a direct ablation baseline.** It has
   `lambda_nce=2.0` and interval 1, whereas the 200 epoch Exp1/Exp4 sweep has
   `lambda_nce=1.0` and interval 4. Its best test PSNR is 20.6273 dB at epoch
   185; this is descriptive context, not an isolated decorrelation comparison.

## Retained And Removed Artifacts

Retain the six Exp1/Exp4 multi-seed directories, the interval-1 final test, and
the existing baselines. Remove only the four 50 epoch quick-test directories:
`expander_decorr_quick_test`, `expander_decorr_quick_test_2`,
`expander_decorr_quick_test_3`, and `expander_decorr_quick_test_4`. Their role
was initial weight screening and they are superseded by the completed 200 epoch
experiments. This recovers approximately 1.46 GiB.