# DarkGhost-ESPNet: báo cáo tổng hợp train - PC - board

Báo cáo này chỉ dùng artifact có sẵn trong project tại thời điểm 26/07/2026.

## Kết quả chính

- Best research checkpoint: `plateau_score_best_monitor`, PSNR 19.6221, SSIM 0.843353 trên `splits/lol_test.txt` (15 ảnh).
- Current deployed ONNX: `stm32/onnx/ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx`.
- PC-board equivalence: layout `raw_nchw`, exact 98.09%, MAE lượng tử 0.022/255, cosine 1.000000.
- Board pipeline hiện tại: inference mean 170.3256 ms, total mean 191.5581 ms, FPS mean 4.9070.

## Traceability

- Báo cáo Word: `D:\LLIE_Project\reports\benchmarks\final_comprehensive_20260726\bao_cao_hoan_chinh_darkghost_espnet_train_pc_board_20260726.docx`
- Contact sheet best checkpoint: `D:\LLIE_Project\reports\figures\final_comprehensive_20260726\best_research_checkpoint_contact_sheet.png`