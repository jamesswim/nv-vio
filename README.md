# VINS-Fusion (Taipei Tech Master's Thesis Enhanced Version)

## An Optimization-based Multi-Sensor State Estimator for Downward-Looking UAV Applications with a Dynamic PnP Strategy

<p align="center">
  <img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/support_files/image/vins_logo.png" width="55%" alt="VINS-Fusion Logo">
  <img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/support_files/image/kitti.png" width="34%" alt="KITTI Dataset Example">
</p>

This project is an enhanced version of the original [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) developed as part of a Master's thesis research at the Department of Computer Science and Information Engineering, National Taipei University of Technology.

The core enhancement is a **motion-aware dynamic pose estimation strategy**, designed to address the unique challenges faced by drones using downward-looking cameras for localization. This strategy significantly improves the robustness and accuracy of localization, especially during critical initialization phases (e.g., vertical take-off), where the coplanarity of image features often degrades the performance of standard VIO systems.

The complete source code and custom datasets are publicly available at: **https://github.com/jamesswim/VINS-Fusion**

**Author:** [鄭泳禎 (Yung-Chen Cheng)](https://github.com/jamesswim)  
**Advisor:** Dr. Hui-Yung Lin  
**Affiliation:** Department of Computer Science and Information Engineering, National Taipei University of Technology (Taipei Tech)

### **Thesis Citation**
If you use this enhanced version of VINS-Fusion in your academic research, please cite the following Master's thesis:

```bibtex
@mastersthesis{Cheng2025VINS,
  author  = {Cheng, Yung-Chen},
  title   = {Implementation of a Visual-Inertial Odometry for Drones with Downward-Looking Aerial Images},
  school  = {National Taipei University of Technology},
  year    = {2025},
  month   = {7}
}
```

### **Core Features**
This version inherits all the functionalities of the original VINS-Fusion and adds:
* **Motion-Aware Dynamic PnP Solver Selection**：Automatically analyzes the drone's motion trends to intelligently switch between the general-purpose `EPnP` solver and the `IPPE` solver, which is specialized for planar scenes. This mechanism enhances robustness during vertical flight and improves accuracy across diverse scenarios.
* **Downward-Looking Stereo Visual-Inertial Dataset**：To address the lack of research data in this specific area, this work contributes a new public dataset. It was collected at an altitude of 25 to 30 meters and covers various outdoor scenes to promote related research.
---

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
* Ubuntu 64-bit 16.04 or 18.04。
* ROS Kinetic or Melodic。[ROS 安裝指引](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**
* Please follow the [Ceres 安裝指引](http://ceres-solver.org/installation.html) for installation.

## 2. Build Instructions
The build process is identical to the original VINS-Fusion.
```bash
cd ~/catkin_ws/src
git clone https://github.com/jamesswim/VINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Reproducing Thesis Experiments
This section provides detailed instructions on how to reproduce the experimental results from Chapter 5 of the thesis.

### 3.1 Key Parameters of the Dynamic Strategy
The core of this research—the dynamic solver selection logic—is controlled by two key parameters defined in the code. Their values are empirical and were set based on experimental results.
* **`MIN_MOTION_THRESHOLD`**: Set to `0.01` (meters). This threshold is used to filter out minor sensor noise, ensuring that motion pattern analysis is triggered only upon significant displacement.
* **`VERTICAL_MOTION_RATIO_THRESHOLD`**: Set to `0.85`. When the ratio of vertical displacement to total displacement exceeds this value, the system identifies the current mode as "vertical take-off/landing" and switches to the `IPPE` solver.

### 3.2 Running Comparison with Mainstream Algorithms
To reproduce the comparison results from Section 5.1, you can run the corresponding launch files on the public datasets used in the thesis.

**Running on the "FGI Campus Dataset":**
```bash
# Open four separate terminals to run rviz, the VINS node, and rosbag
roslaunch vins vins_rviz.launch
rosrun vins vins_node path/to/your_config/FGI_Masala/FGI_Masala_Stereo_mono8_config.yaml
rosrun loop_fusion loop_fusion_node path/to/your_config/FGI_Masala/FGI_Masala_Stereo_mono8_config.yaml  
rosbag play YOUR_DATASET_FOLDER/40_2.bag
```

**Running on the "Diverse Outdoor Scenes Dataset":**
```bash
# Open four separate terminals to run rviz, the VINS node, and rosbag
roslaunch vins vins_rviz.launch
rosrun vins vins_node path/to/your_config/low_altitude/nav_stereo_imu.yaml
rosrun loop_fusion loop_fusion_node path/to/your_config/low_altitude/nav_stereo_imu.yaml
rosbag play YOUR_DATASET_FOLDER/0628_50_5.bag
```

### 3.3 Running the Ablation Study
To validate the effectiveness of the dynamic strategy, you can reproduce the ablation study from Section 5.2. This requires modifying parts of the code to force the use of a specific solver.

The dynamic selection logic is primarily implemented in the `FeatureManager::initFramePoseByPnP` function within  `vins_estimator/src/feature_manager.cpp`.
1.  **Baseline (Fixed EPnP)**：This is the default behavior of the original VINS-Fusion.
2.  **Fixed IPPE Group**：Modify the code to bypass the motion pattern check, forcing the  `solvePoseByPnP` function to always use the `IPPE` solver. This will demonstrate the instability of a single specialized solver when the planarity assumption is not met.
3.  **Our Method (Dynamic Switching)**： The default behavior of this project, which dynamically selects between `IPPE`(for vertical motion) and `EPnP` (for non-vertical motion) based on the motion pattern.

By running these three configurations on the "FGI Campus Dataset," you will be able to reproduce the trajectory and error analysis results from Figures 5.7, 5.8, 5.9 and Tables 5.5, 5.6 in the thesis.

## 4. Using Our Custom Taipei Tech Dataset
This research contributes a new downward-looking UAV dataset, collected at locations including:
* Jiangzicui Riverside Park
* Yinyue Park (Music Park)
* National Taipei University of Technology (Taipei Tech) Athletic Field

The dataset (including sensor data `.bag`, ground truth trajectory `.kml` , and calibration files  `.yaml`) is available for download.

## 5. Original Documentation & Acknowledgements
For instructions on running the system on other datasets (e.g., EuRoC, KITTI) or with your own equipment, please refer to the documentation of the original  [VINS-Fusion 專案](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) .

This project uses [Ceres Solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop closure detection. We thank the original authors of VINS-Fusion for their outstanding work.

## 6. License
The source code of this project is released under the [GPLv3](http://www.gnu.org/licenses/) license.
