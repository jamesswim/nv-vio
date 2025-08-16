# VINS-Fusion (北科大碩士論文改良版)

## 一套具備動態PnP策略、針對下視角無人機應用的多感測器狀態估計器

<p align="center">
  <img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/support_files/image/vins_logo.png" width="55%" alt="VINS-Fusion Logo">
  <img src="https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/raw/master/support_files/image/kitti.png" width="34%" alt="KITTI Dataset Example">
</p>

本專案為原始 [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 的改良版本，是國立臺北科技大學資訊工程系碩士論文的研究成果。

核心改良項目為一套**基於運動感知的動態位姿估計策略**，旨在解決無人機採用下視角攝影機進行定位時的獨特挑戰。此策略顯著提升了定位的穩健性與精準度，尤其是在系統初始化的關鍵階段（例如：垂直起飛），此時影像特徵的共平面性常會導致標準VIO系統的性能下降。

完整的程式碼及自建資料集皆公開於：**https://github.com/jamesswim/VINS-Fusion**

**作者：** [鄭泳禎 (Yung-Chen Cheng)](https://github.com/jamesswim)  
**指導教授：** 林惠勇 博士  
**單位：** 國立臺北科技大學 資訊工程系

### **論文引用**
若您在學術研究中使用此改良版 VINS-Fusion，請引用以下碩士論文：

```bibtex
@mastersthesis{Cheng2025VINS,
  author  = {鄭泳禎},
  title   = {利用下視角空拍影像之無人機視覺里程計建置},
  school  = {國立臺北科技大學},
  year    = {2025},
  month   = {7}
}
```

### **核心功能**
此版本繼承了原始 VINS-Fusion 的所有功能，並新增：
* **運動感知之動態 PnP 求解器選擇**：能自動分析無人機的運動趨勢，在通用型的 `EPnP` 求解器與針對平面場景特化的 `IPPE` 求解器之間進行智慧切換。此機制強化了垂直飛行時的穩健性，並提升了在多樣化場景中的精準度。
* **下視角雙目視覺慣性資料集**：為彌補此領域研究資料的不足，本研究建立並公開了一套全新的資料集。該資料集於 25 至 30 公尺高度蒐集，涵蓋多樣化的戶外場景，以促進相關研究發展。

---

## 1. 環境需求
### 1.1 **Ubuntu** 與 **ROS**
* Ubuntu 64-bit 16.04 或 18.04。
* ROS Kinetic 或 Melodic。[ROS 安裝指引](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**
* 請遵循 [Ceres 安裝指引](http://ceres-solver.org/installation.html) 進行安裝。

## 2. 編譯說明
編譯流程與原始 VINS-Fusion 相同。
```bash
cd ~/catkin_ws/src
git clone https://github.com/jamesswim/VINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. 重現論文實驗
本章節提供如何重現論文第五章實驗結果的詳細說明。

### 3.1 動態策略之關鍵參數
本研究的核心—動態求解器選擇邏輯，由以下兩個定義在程式碼中的關鍵參數控制。其數值是根據實驗結果設定的經驗值。
* **`MIN_MOTION_THRESHOLD`**: 設為 `0.01` (公尺)。此閾值用於過濾感測器微小雜訊，確保僅在發生顯著位移時才觸發運動模式分析。
* **`VERTICAL_MOTION_RATIO_THRESHOLD`**: 設為 `0.85`。當垂直位移佔總位移的比例超過此數值時，系統會判定當前為「垂直起降」模式，並切換至 `IPPE` 求解器。

### 3.2 執行與主流演算法之比較
若要重現 5.1 節的比較結果，您可以在論文所使用的公開資料集上執行對應的 launch 文件。

**於「FGI 建築周遭資料集」上執行：**
```bash
# 分別開啟四個終端機執行 rviz, VINS 節點, 以及 rosbag
roslaunch vins vins_rviz.launch
rosrun vins vins_node path/to/your_config/FGI_Masala/FGI_Masala_Stereo_mono8_config.yaml
rosrun loop_fusion loop_fusion_node path/to/your_config/FGI_Masala/FGI_Masala_Stereo_mono8_config.yaml  
rosbag play YOUR_DATASET_FOLDER/40_2.bag
```

**於「多樣化戶外場景資料集」上執行：**
```bash
# 分別開啟四個終端機執行 rviz, VINS 節點, 以及 rosbag
roslaunch vins vins_rviz.launch
rosrun vins vins_node path/to/your_config/low_altitude/nav_stereo_imu.yaml
rosrun loop_fusion loop_fusion_node path/to/your_config/low_altitude/nav_stereo_imu.yaml
rosbag play YOUR_DATASET_FOLDER/0628_50_5.bag
```

### 3.3 執行消融實驗 (Ablation Study)
為驗證動態策略的有效性，您可以重現 5.2 節的消融實驗。這需要修改部分程式碼，以強制使用特定的求解器。

動態選擇的邏輯主要實作於 `vins_estimator/src/feature_manager.cpp` 的 `FeatureManager::initFramePoseByPnP` 函數中。
1.  **基準組 (固定使用 EPnP)**：此為原始 VINS-Fusion 的行為。
2.  **固定使用 IPPE 組**：修改程式碼，繞過運動模式判斷，強制 `solvePoseByPnP` 函數總是使用 `IPPE` 求解器。這將會展示在不滿足平面假設時，單一特化求解器的不穩定性。
3.  **本論文方法 (動態切換)**：本專案的預設行為，即根據運動模式在 `IPPE` (垂直運動) 和 `EPnP` (非垂直運動) 之間動態選擇。

透過在「FGI 建築周遭資料集」上執行這三種設定，您將能夠重現論文中圖 5.7、5.8、5.9 以及表 5.5、5.6 的軌跡與誤差分析結果。

## 4. 使用自建之北科大資料集
本研究貢獻了一套全新的下視角無人機資料集，採集地點包含：
* 江子翠河濱公園
* 音樂公園
* 國立臺北科技大學操場

資料集（包含感測器數據 `.bag`、真實軌跡 `.kml` 及校正檔 `.yaml`）

## 5. 原始文件與致謝
關於在其他資料集（如 EuRoC, KITTI）或您自己的設備上運行的說明，請參考原始 [VINS-Fusion 專案](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 的文件。

本專案使用 [Ceres Solver](http://ceres-solver.org/) 進行非線性優化，並使用 [DBoW2](https://github.com/dorian3d/DBoW2) 進行迴圈檢測。感謝 VINS-Fusion 原始作者們的傑出工作。

## 6. 授權條款
本專案之原始碼依 [GPLv3](http://www.gnu.org/licenses/) 授權條款釋出。
