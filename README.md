# VINS-Fusion: Enhanced for Downward-Looking UAV Localization
## An optimization-based multi-sensor state estimator

<img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/support_files/image/vins_logo.png" width = 55% height = 55% div align=left />
<img src="support_files/image/your_uav_photo.png" width = 34% height = 34% div align=center /> 
This repository is an enhanced version of the original [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). The core improvement is the implementation of a **dynamic pose estimation strategy** specifically designed to enhance the robustness and accuracy of **downward-looking UAVs**.

This modification addresses a key challenge in visual-inertial odometry (VIO): the performance degradation during the critical initialization phase when a UAV performs vertical take-off over planar or texture-less ground. By dynamically perceiving the UAV's motion patterns, our system adaptively switches between different pose estimation solvers to achieve superior performance without requiring additional hardware.

### Core Contributions of This Fork

* **Motion-Aware Dynamic Pose Estimation:** Implemented a novel strategy that analyzes the UAV's real-time motion trends. It intelligently switches to the **IPPE** solver during vertical motion to leverage planar constraints, and reverts to the robust **EPnP** solver for general, non-vertical motion. This significantly improves initialization stability and final accuracy.

* **Enhanced Robustness for Downward-Looking Scenarios:** Solves the pose estimation instability caused by the co-planarity of visual features during vertical take-off, a common failure point for many VIO systems.

* **A New Public Dataset for Downward-Looking VIO:** To facilitate further research in this area, we have built and publicly released a new **NTUT Downward-Looking UAV Dataset**. This dataset, available at the project's GitHub repository, provides valuable resources for future downward-looking VIO research.

### Citing This Work

If you use the enhancements or the dataset from this repository in your academic research, please cite the following thesis:

```bibtex
@mastersthesis{Cheng2025UAV,
  author  = {Cheng, Yung-Chen (鄭泳禎)},
  title   = {Development of a UAV Visual Odometry System Based on Downward-looking Aerial Imagery (利用下視角空拍影像之無人機視覺里程計建置)},
  school  = {National Taipei University of Technology (國立臺北科技大學)},
  year    = {2025},
  month   = {7},
  advisor = {Lin, Hui-Yung (林惠勇)},
}
