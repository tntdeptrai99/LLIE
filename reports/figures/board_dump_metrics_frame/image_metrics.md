# Board Frame Image Metrics

Contact sheet: `D:\LLIE_Project\reports\figures\board_dump_metrics_frame\board_input_vs_ai_output_contact_x4.png`
Histogram: `D:\LLIE_Project\reports\figures\board_dump_metrics_frame\luma_histogram_input_vs_ai_output.png`

| metric | input/preprocess | AI output |
|---|---:|---:|
| brightness_mean | 38.852962 | 101.951805 |
| contrast_std | 31.218187 | 48.144119 |
| contrast_p01_p99 | 148.695998 | 203.760553 |
| saturation_mean | 0.501802 | 0.260975 |
| sharpness_laplacian_abs_mean | 13.807174 | 22.006273 |
| sharpness_laplacian_variance | 483.002991 | 1020.672729 |
| clip_0_ratio_rgb | 0.119936 | 0.000000 |
| clip_255_ratio_rgb | 0.000000 | 0.003906 |

Notes: brightness/contrast/sharpness use luma; saturation uses HSV S channel; clipping ratios are fractions of RGB samples.