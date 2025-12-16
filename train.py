import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time

# 1. 設定參數
BATCH_SIZE = 32  # 您的 5060 Ti 夠強，可以設 32 或 64 加快速度
LEARNING_RATE = 0.001
EPOCHS = 20      # 建議練 20 輪
DATA_DIR = './traindata' # 指向您的手指資料夾
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f"🚀 使用裝置: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   顯示卡型號: {torch.cuda.get_device_name(0)}")

    # 2. 圖片預處理 (ResNet 標準處理流程)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),       # ResNet 需要 224x224
        transforms.RandomHorizontalFlip(),   # [新增] 資料增強：隨機左右翻轉
        transforms.RandomRotation(15),       # [新增] 資料增強：隨機旋轉
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 讀取資料
    if not os.path.exists(DATA_DIR):
        print(f"錯誤：找不到 {DATA_DIR} 資料夾！")
        return

    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    
    # 自動偵測類別數量
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"📂 偵測到的類別: {class_names} (共 {num_classes} 類)")

    # 分割 80% 訓練, 20% 驗證
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📊 訓練圖片: {len(train_data)} 張, 驗證圖片: {len(val_data)} 張")

    # 4. 建立模型 (使用預訓練的 ResNet18)
    # weights='IMAGENET1K_V1' 是新版寫法，取代 pretrained=True
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 修改最後一層全連接層 (Fully Connected Layer)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # 修改輸出為 6 (0~5)
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 開始訓練
    train_acc_history = []
    val_acc_history = []
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- 訓練階段 ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        train_acc_history.append(epoch_acc)

        # --- 驗證階段 (新增的部分，檢查考試成績) ---
        model.eval() # 設定為評估模式
        val_correct = 0
        val_total = 0
        with torch.no_grad(): # 驗證時不需要算梯度，省記憶體
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_acc_history.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | 訓練 Acc: {epoch_acc:.2f}% | 驗證 Acc: {val_acc:.2f}%")

    time_elapsed = time.time() - start_time
    print(f"\n✅ 訓練完成！耗時: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒")

    # 6. 儲存模型
    torch.save(model.state_dict(), 'finger_model_pytorch.pth')
    print("💾 模型已儲存為 finger_model_pytorch.pth")

    # 7. 畫圖
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_chart_pytorch.png')
    print("📈 圖表已儲存為 result_chart_pytorch.png")

if __name__ == '__main__':
    train_model()