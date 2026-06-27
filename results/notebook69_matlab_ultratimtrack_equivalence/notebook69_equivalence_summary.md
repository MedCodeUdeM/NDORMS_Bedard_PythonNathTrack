# Notebook 69 MATLAB-vs-Python equivalence summary

Compared 2666 matched frames from `UltraTimTrack_test.mp4`.
MATLAB rows were aligned to Python/video frames using Python offset 1.
Equivalence margins were +/-5 mm for fascicle length and +/-5 deg for fascicle angle.

Both primary endpoints met the pre-specified mean-equivalence criterion.

L / fascicle length: Python minus MATLAB bias was -2.095 mm (moving-block bootstrap 95% CI -2.578 to -1.767 mm), RMSE 2.867 mm, Lin's CCC 0.955; this met the pre-specified +/-5 mm mean-equivalence criterion.
phi / fascicle angle: Python minus MATLAB bias was 1.192 deg (moving-block bootstrap 95% CI 0.994 to 1.438 deg), RMSE 1.599 deg, Lin's CCC 0.947; this met the pre-specified +/-5 deg mean-equivalence criterion.
PEN / pennation angle: Python minus MATLAB bias was 1.109 deg (moving-block bootstrap 95% CI 0.925 to 1.342 deg), RMSE 1.524 deg, Lin's CCC 0.945; this met the pre-specified +/-5 deg mean-equivalence criterion.

Requested length plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook69_matlab_ultratimtrack_equivalence/fascicle_length_matlab_vs_python_over_time.png`
Requested angle plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook69_matlab_ultratimtrack_equivalence/fascicle_angle_matlab_vs_python_over_time.png`
Aligned data CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook69_matlab_ultratimtrack_equivalence/aligned_matlab_python_fascicle_outputs.csv`
Statistics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook69_matlab_ultratimtrack_equivalence/matlab_python_equivalence_statistics.csv`