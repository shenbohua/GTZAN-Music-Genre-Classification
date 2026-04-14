import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

def evaluate(model, loader, criterion):
    model.eval()
    v_loss, v_correct = 0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            out = model(x)
            v_loss += criterion(out, y).item() * x.size(0)
            v_correct += (out.argmax(1) == y).sum().item()
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
    return v_loss / len(loader.dataset), v_correct / len(loader.dataset), all_labels, all_probs

def run_training(model, train_loader, val_loader, optimizer, epochs, name, use_patience=False):
    latest_ckpt_path = os.path.join(Config.SAVE_DIR, f"{name}_latest.pth")
    best_ckpt_path = os.path.join(Config.SAVE_DIR, f"{name}_best.pth")
    history_path = os.path.join(Config.SAVE_DIR, f"{name}_history.json")

    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    model.to(Config.DEVICE)

    # 1. 检测检查点，尝试恢复训练进度
    if os.path.exists(latest_ckpt_path):
        print(f"\n🔄 发现本地进度 [{name}]，尝试断点续训...")
        checkpoint = torch.load(latest_ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        
        # 如果已经跑完了预定目标，直接加载最优模型并跳过循环
        if start_epoch >= epochs:
            print(f"✅ 模型 {name} 已完成 {epochs}/{epochs} 轮训练，直接加载最佳权重。")
            if os.path.exists(best_ckpt_path):
                model.load_state_dict(torch.load(best_ckpt_path, map_location=Config.DEVICE))
            return model
        else:
            print(f"🚀 成功恢复状态！从 Epoch {start_epoch+1}/{epochs} 继续训练...")
    else:
        print(f"\n>> 🆕 开始全新实验: {name} (目标轮次: {epochs})")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss, t_correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (out.argmax(1) == y).sum().item()
        
        v_loss, v_correct, _, _ = evaluate(model, val_loader, criterion)
        
        metrics = {
            'train_loss': t_loss/len(train_loader.dataset), 'train_acc': t_correct/len(train_loader.dataset),
            'val_loss': v_loss, 'val_acc': v_correct
        }
        for k, v in metrics.items(): history[k].append(v)

        # 2. 判断并保存最佳模型权重
        if metrics['val_acc'] > best_acc:
            best_acc = metrics['val_acc']
            torch.save(model.state_dict(), best_ckpt_path)
            patience_counter = 0
        else: patience_counter += 1

        # 3. 每一个 Epoch 保存一次 最新进度 (包含优化器、历史记录等完整状态)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_acc': best_acc
        }, latest_ckpt_path)

        # 实时保存 history JSON，防止最后没写入
        with open(history_path, 'w') as f: json.dump(history, f)

        print(f"Epoch {epoch+1:03d}/{epochs}: T-Acc {metrics['train_acc']:.4f} | V-Acc {metrics['val_acc']:.4f}")
        
        if use_patience and patience_counter >= 10: 
            print("触发 Early Stopping")
            break
            
    # 训练结束后，返回最佳模型
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=Config.DEVICE))
    return model