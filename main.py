import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import os

# --- 設定參數 ---
MODEL_PATH = 'finger_model_pytorch.pth' # 請確認檔名與訓練時一致
INPUT_DIR = './input'                   # 測試圖片資料夾
CLASSES = ['0', '1', '2', '3', '4', '5'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY_WIDTH = 800  # 限制顯示視窗寬度，避免圖片太大爆框

def load_trained_model():
    print(f"正在載入模型... (使用裝置: {DEVICE})")
    # 1. 重建模型骨架 (ResNet18)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES)) # 修改輸出層為 6
    
    # 2. 載入權重
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到 {MODEL_PATH}，請先執行 train_torch.py")
        return None
        
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return None

    model = model.to(DEVICE)
    model.eval() # 設定為推論模式 (重要！)
    return model

def predict_from_folder():
    # 檢查 input 資料夾
    if not os.path.exists(INPUT_DIR):
        print(f"錯誤：找不到 {INPUT_DIR} 資料夾")
        return

    # 載入模型
    model = load_trained_model()
    if model is None: return

    # 定義預處理 (必須跟訓練時一樣)
    # OpenCV 讀進來是 BGR，轉成 PIL (RGB) 後再做 Transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 取得圖片列表
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    
    if not image_files:
        print(f"{INPUT_DIR} 內沒有圖片。")
        return

    print(f"找到 {len(image_files)} 張圖片。按 '任意鍵' 下一張，按 'q' 離開。")

    for filename in image_files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        # 1. 使用 OpenCV 讀取圖片
        img_bgr = cv2.imread(file_path)
        if img_bgr is None: continue

        # --- 顯示用的縮放處理 (避免圖太大) ---
        h, w = img_bgr.shape[:2]
        if w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / w
            new_h = int(h * scale)
            display_img = cv2.resize(img_bgr, (DISPLAY_WIDTH, new_h))
        else:
            display_img = img_bgr.copy()

        # --- 2. 轉換格式給 PyTorch 模型 ---
        # OpenCV (BGR) -> RGB -> PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 預處理 -> 增加 Batch 維度 [1, 3, 224, 224] -> 搬到 GPU/CPU
        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        # --- 3. 推論 ---
        with torch.no_grad():
            outputs = model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 取得最高分的類別
        top_prob, top_class = torch.max(probabilities, 0)
        predicted_label = CLASSES[top_class.item()]
        confidence = top_prob.item() * 100

        # --- 4. 顯示結果 ---
        print(f"檔案: {filename} -> {predicted_label} ({confidence:.2f}%)")
        
        # 根據信心度變色
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        
        text = f"Pred: {predicted_label} ({confidence:.1f}%)"
        cv2.putText(display_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('PyTorch Prediction', display_img)

        # 等待按鍵
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_from_folder()