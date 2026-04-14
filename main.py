import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd

from config import Config
from data_loader import get_image_dataloaders, get_audio_dataloaders
from models import Net1, Net2, Net3, Net4, Net5, Net6, Generator, Discriminator
from engine import run_training
from utils import evaluate_and_visualize_model, plot_training_history, plot_global_comparison

if __name__ == "__main__":
    global_results = {}
    
    # -----------------------------------
    # A. 图像任务：Net1 ~ Net4
    # -----------------------------------
    img_train_loader, img_val_loader, img_test_loader, _ = get_image_dataloaders()

    networks = [Net1, Net2, Net3, Net4]
    for net_cls in networks:
        for epoch_count in [50, 100]: 
            exp_name = f"{net_cls.__name__}_{epoch_count}ep"
            model = net_cls()
            opt = optim.RMSprop(model.parameters(), lr=1e-4) if net_cls == Net4 else optim.Adam(model.parameters(), lr=1e-3)
            
            # 这里的引擎内部自带检测：如果本地存在进度或已完成，它会自动恢复或跳过
            trained_model = run_training(model, img_train_loader, img_val_loader, opt, epoch_count, exp_name, use_patience=False)
            
            metrics = evaluate_and_visualize_model(trained_model, img_test_loader, exp_name)
            global_results[exp_name] = metrics
            
            plot_training_history(exp_name)

    # -----------------------------------
    # B. 音频任务：Net5 与 Net6 (GAN)
    # -----------------------------------
    audio_train_loader, audio_val_loader, audio_test_loader, audio_train_ds = get_audio_dataloaders()

    # 1. 训练 Net5
    net5_model = Net5()
    trained_net5 = run_training(net5_model, audio_train_loader, audio_val_loader, optim.Adam(net5_model.parameters(), lr=1e-3), 50, "Net5_LSTM", use_patience=True)
    metrics_net5 = evaluate_and_visualize_model(trained_net5, audio_test_loader, "Net5_LSTM")
    global_results["Net5_LSTM"] = metrics_net5
    plot_training_history("Net5_LSTM")
    # 2. GAN 对抗训练与数据增强 (带本地保存机制)
    gan_gen_path = os.path.join(Config.SAVE_DIR, "GAN_Generator_best.pth")
    generator = Generator().to(Config.DEVICE)
    
    if os.path.exists(gan_gen_path):
        print("\n🔄 发现已训练好的 GAN 生成器，直接加载，跳过 GAN 训练环节...")
        generator.load_state_dict(torch.load(gan_gen_path, map_location=Config.DEVICE))
    else:
        print("\n>> 🆕 正在训练 GAN 生成器以产出音频增强数据...")
        discriminator = Discriminator().to(Config.DEVICE)
        g_opt = optim.Adam(generator.parameters(), lr=0.0002)
        d_opt = optim.Adam(discriminator.parameters(), lr=0.0002)
        criterion_gan = nn.BCELoss()
        
        for epoch in range(15):
            for x, _ in audio_train_loader:
                x = x.to(Config.DEVICE)
                b_size = x.size(0)
                d_opt.zero_grad()
                d_real_loss = criterion_gan(discriminator(x), torch.ones(b_size, 1).to(Config.DEVICE))
                fake_x = generator(torch.randn(b_size, 100).to(Config.DEVICE))
                d_fake_loss = criterion_gan(discriminator(fake_x.detach()), torch.zeros(b_size, 1).to(Config.DEVICE))
                (d_real_loss + d_fake_loss).backward()
                d_opt.step()
                g_opt.zero_grad()
                criterion_gan(discriminator(fake_x), torch.ones(b_size, 1).to(Config.DEVICE)).backward()
                g_opt.step()
        # 训练完成后保存生成器，方便下次直接跳过
        torch.save(generator.state_dict(), gan_gen_path)
        print("✅ GAN 训练完成并已保存至本地。")

    print("\n>> 正在使用 GAN 生成伪造特征以扩充训练集...")
    num_fake = len(audio_train_ds)
    generator.eval()
    with torch.no_grad(): fake_mfcc = generator(torch.randn(num_fake, 100).to(Config.DEVICE)).cpu()
    
    real_mfcc = torch.stack([audio_train_ds[i][0] for i in range(num_fake)])
    real_labels = torch.tensor([audio_train_ds[i][1] for i in range(num_fake)])
    aug_train_dataset = TensorDataset(torch.cat([real_mfcc, fake_mfcc], dim=0), torch.cat([real_labels, real_labels], dim=0))
    aug_train_loader = DataLoader(aug_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

    # 3. 训练 Net6
    net6_model = Net6()
    trained_net6 = run_training(net6_model, aug_train_loader, audio_val_loader, optim.Adam(net6_model.parameters(), lr=1e-3), 50, "Net6_LSTM_GAN", use_patience=True)
    metrics_net6 = evaluate_and_visualize_model(trained_net6, audio_test_loader, "Net6_LSTM_GAN")
    global_results["Net6_LSTM_GAN"] = metrics_net6
    plot_training_history("Net6_LSTM_GAN")
    # -----------------------------------
    # C. 全局结果汇总与多格式持久化导出
    # -----------------------------------
    plot_global_comparison(global_results)
    
    # 导出 JSON 格式数据
    final_json_path = os.path.join(Config.SAVE_DIR, "Final_Global_Metrics.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(global_results, f, indent=4, ensure_ascii=False)
        
    # 导出 CSV 格式数据，极大地提高写论文制作表格的效率
    final_csv_path = os.path.join(Config.SAVE_DIR, "Final_Global_Metrics.csv")
    df_results = pd.DataFrame.from_dict(global_results, orient='index')
    df_results.index.name = 'Model_Architecture'
    df_results.to_csv(final_csv_path)

    print(f"\n🎉 完美收官！所有检查点、折线图、混淆矩阵、以及全局指标文件均已安全落盘至目录：\n 📁 {os.path.abspath(Config.SAVE_DIR)}")