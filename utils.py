import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score, log_loss
from sklearn.preprocessing import label_binarize
from config import Config
import pandas as pd

def plot_roc_curve(all_labels, all_probs, exp_name):
    labels_bin = label_binarize(all_labels, classes=range(Config.NUM_CLASSES))
    all_probs = np.array(all_probs)
    
    plt.figure(figsize=(10, 8))
    for i in range(Config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{Config.CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {exp_name}')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(Config.SAVE_DIR, f"{exp_name}_ROC.png"), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_and_visualize_model(model, test_loader, exp_name):
    """
    计算Accuracy, Macro Precision, Macro Recall, Macro F1, Kappa, LogLoss。
    绘制混淆矩阵并美化终端打印。
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            out = model(x)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(1)

            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算各种高级评估指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    lloss = log_loss(all_labels, all_probs)

    # 绘制 ROC 曲线
    plot_roc_curve(all_labels, all_probs, exp_name)

    # 绘制并保存混淆矩阵 (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    plt.title(f'Confusion Matrix - {exp_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, f"{exp_name}_ConfusionMatrix.png"), dpi=300)
    plt.close()

    # 控制台美化打印
    print(f"\n{'='*50}")
    print(f"📊 模型 {exp_name} 测试集综合评估报告")
    print(f"{'='*50}")
    print(f"  ▶ Accuracy        : {acc:.4f}")
    print(f"  ▶ Macro Precision : {prec:.4f}")
    print(f"  ▶ Macro Recall    : {rec:.4f}")
    print(f"  ▶ Macro F1 Score  : {f1:.4f}")
    print(f"  ▶ Cohen's Kappa   : {kappa:.4f}")
    print(f"  ▶ Log Loss        : {lloss:.4f}")
    print(f"{'='*50}\n")

    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'Kappa': kappa}

def plot_global_comparison(results_dict):
    """汇总各个网络的指标，生成折线图和柱状图进行对比"""
    if not results_dict: return
    models = list(results_dict.keys())
    metrics_to_plot = ['Accuracy', 'F1', 'Precision', 'Recall', 'Kappa']

    # 1. 绘制综合柱状图 (Bar Chart)
    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results_dict[m][metric] for m in models]
        ax.bar(x + i*width - (len(metrics_to_plot)*width/2), values, width, label=metric)

    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Global Models Performance Comparison (Bar Chart)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, "Global_Comparison_Bar.png"), dpi=300)
    plt.close()

    # 2. 绘制综合折线图 (Line Chart)
    plt.figure(figsize=(12, 6))
    for metric in metrics_to_plot:
        values = [results_dict[m][metric] for m in models]
        plt.plot(models, values, marker='o', linewidth=2, markersize=8, label=metric)

    plt.title('Performance Trend Across Models (Line Chart)', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, "Global_Comparison_Line.png"), dpi=300)
    plt.close()
    print("📈 全局对比折线图与柱状图已生成！")

def plot_training_history(exp_name):
    """
    读取本地的 history.json 文件，绘制并保存训练过程中的 Loss 和 Accuracy 曲线。
    用于直观诊断模型是否出现过拟合。
    """
    history_path = os.path.join(Config.SAVE_DIR, f"{exp_name}_history.json")
    if not os.path.exists(history_path):
        print(f"⚠️ 未找到 {exp_name} 的历史记录文件，无法绘制学习曲线。")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建 1行2列 的子图 (Loss 和 Accuracy并排)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制 Loss 曲线 (蓝色实线代表训练集，红色虚线代表验证集)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title(f'Training and Validation Loss - {exp_name}', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 绘制 Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'Training and Validation Accuracy - {exp_name}', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, f"{exp_name}_LearningCurve.png"), dpi=300)
    plt.close()