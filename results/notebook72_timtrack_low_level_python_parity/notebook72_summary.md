# Notebook 72 summary

MATLAB export: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook72_timtrack_low_level_matlab_export/matlab_timtrack_low_level_UltraTimTrack_test.mat`
Frames: [0, 1, 2, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2665]

- Same MATLAB mask -> Python dohough alpha: mean MAE 1.2 deg, worst MAE 6.5 deg.
- Same MATLAB mask -> Python Hough h_by_angle: mean MAE 0 votes, worst MAE 0 votes.
- MATLAB fas_thres + Python get_fasMask -> alpha: mean MAE 1.2 deg, worst MAE 6.5 deg.
- Full independent Python low-level alpha: mean MAE 0.966667 deg, worst MAE 5.5 deg.

Worst binary stages by different-pixel fraction:
- apo_thres: mean diff fraction 0.582631, mean Dice 0.514888, worst Dice 0.502098
- apo_deep: mean diff fraction 0.0768117, mean Dice 0.53655, worst Dice 0.490349
- apo_super: mean diff fraction 0.0730377, mean Dice 0.505432, worst Dice 0.49607
- Emask: mean diff fraction 0.0453617, mean Dice 0.935418, worst Dice 0.906159
- fascicle_masked: mean diff fraction 0.00782935, mean Dice 0.941573, worst Dice 0.913893
- fas_thres: mean diff fraction 0.00627413, mean Dice 0.984891, worst Dice 0.980831
- fas_thres_raw: mean diff fraction 0.00627413, mean Dice 0.984891, worst Dice 0.980831

Worst numeric stages by MAE:
- super_vec: mean MAE 23.1467 px, worst MAE 26.3
- deep_vec: mean MAE 12.8333 px, worst MAE 26
- h_by_angle_full_python: mean MAE 9.27808 votes, worst MAE 12.4685
- Emask_radius: mean MAE 7.83333 px, worst MAE 11.75
- alphas_full_python: mean MAE 5.95667 deg, worst MAE 10.95
- weights_full_python: mean MAE 5.4 votes, worst MAE 8
- alpha_full_python: mean MAE 0.966667 deg, worst MAE 5.5
- fas_filt: mean MAE 0.001087 Frangi response, worst MAE 0.00124426