## Image_Identify_FinalAI_Final_Project/

###  專期簡介

本專案為影像處理概論之期末專題。旨在開發一套能夠透過辨識手指數量 (0-5) 的系統。
為了解決傳統影像處理方法（如膚色偵測、輪廓分析）易受光影變化與複雜背景干擾的問題，本專題採用 **深度學習 (Deep Learning)** 技術進行手勢辨識。

###  資料集與預處理 / Dataset & Preprocessing
#### 資料來源
* **來源**: 自行拍攝手勢影像及網路公開資源。
* **類別**: 共 6 類 (對應手指數量 0, 1, 2, 3, 4, 5)。
* **數量**: 每類約 80 張原始圖片。

#### 資料增強 / Data Augmentation
由於原始資料量較少，為了防止過擬合 (Overfitting) 並提升泛化能力，訓練時採用了以下技術：
* **Resize**: 統一縮放至 224x224。
* **Random Horizontal Flip**: 隨機水平翻轉 (模擬左右手)。
* **Random Rotation**: 隨機旋轉 ±15 度 (模擬手勢角度變化)。
* **Normalization**: 使用 ImageNet 平均值與標準差進行標準化。


### 核心技術

* **Model Architecture**: ResNet18 (Pre-trained on ImageNet)
* **Framework**: PyTorch
* **Computer Vision**: OpenCV
* **Hardware Acceleration**: CUDA (Tested on NVIDIA RTX 5060 Ti)

## 環境需求

Python 3.8+ 及以下套件：

```bash
pip install torch torchvision opencv-python matplotlib pillow numpy
```
  
###  參考資料
 - Mitra, S., & Acharya, T. (2007). Gesture Recognition: A Survey. IEEE Transactions on Systems, Man, and Cybernetics.
 - Keskin, C., et al. (2011). Real Time Hand Pose Estimation Using Depth Sensors. ICCV.
 - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
 - PyTorch Documentation. https://pytorch.org/
 - OpenCV Documentation. https://docs.opencv.org/

###  專期結構

    ├── traindata/                 # 訓練資料集
    ├── output/                    # 預測結果輸出
    ├── input/                     # 測試圖片
    ├── train.py                   # 訓練程式
    ├── main.py                    # 判別用程式
    └── finger_model_pytorch.pth   # 模型檔




