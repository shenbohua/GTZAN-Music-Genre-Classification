import os
import numpy as np
import torch
import wave
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchaudio
from config import Config

# 图像数据集
class GTZANImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_cache = [] 
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print("🔄 正在将图片预处理并加载到内存中...")
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    label = self.class_to_idx[cls_name]
                    
                    image = Image.open(img_path).convert('RGB')
                    image = self._remove_borders(image)
                    if self.transform: 
                        image = self.transform(image)
                        
                    self.data_cache.append(image)
                    self.labels.append(label)
        print(f"✅ 图片加载完成！共缓存 {len(self.data_cache)} 张图像。")

    def __len__(self): return len(self.data_cache)

    def _remove_borders(self, image):
        img_np = np.array(image)
        non_white = np.where(img_np < 240)
        if len(non_white[0]) == 0: return image 
        y_min, y_max = non_white[0].min(), non_white[0].max()
        x_min, x_max = non_white[1].min(), non_white[1].max()
        return Image.fromarray(img_np[y_min:y_max+1, x_min:x_max+1, :])
    
    def __getitem__(self, idx):
        return self.data_cache[idx], self.labels[idx]

# 图像加载器
def get_image_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE), # 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = GTZANImageDataset(Config.IMG_DATA_DIR, transform)
    
    total = len(full_dataset)
    train_sz = int(0.7 * total)
    val_sz = int(0.2 * total)
    test_sz = total - train_sz - val_sz # [cite: 26]
    
    gen = torch.Generator().manual_seed(Config.SEED)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_sz, val_sz, test_sz], generator=gen)
    
    loader_kwargs = {'batch_size': Config.BATCH_SIZE, 'num_workers': 12, 'pin_memory': True}
    return (DataLoader(train_ds, shuffle=True, **loader_kwargs),
            DataLoader(val_ds, shuffle=False, **loader_kwargs),
            DataLoader(test_ds, shuffle=False, **loader_kwargs),
            test_ds)

# 音频数据集
class GTZANAudioDataset(Dataset):
    def __init__(self, root_dir, max_len=1300):
        self.root_dir = root_dir
        self.max_len = max_len
        self.data_cache = []
        self.labels = []
        # self.classes = sorted(os.listdir(root_dir))
        # 🔧 过滤掉隐藏文件夹和非目录文件
        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) 
            and not d.startswith('.')  # 排除隐藏文件夹
        ])
        
        print(f"发现有效类别: {self.classes}")
        
        # 重新映射，确保从0开始
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"类别映射: {self.class_to_idx}")
        
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=22050, n_mfcc=40,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
        )
        
        print("🔄 正在提取音频 MFCC 特征并加载到内存...")
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for file_name in os.listdir(cls_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(cls_dir, file_name)
                        label = self.class_to_idx[cls_name]
                        
                        try:
                            # 尝试用原生 wave 读取
                            with wave.open(file_path, 'rb') as wf:
                                sample_rate = wf.getframerate()
                                n_channels = wf.getnchannels()
                                n_frames = wf.getnframes()
                                audio_data = wf.readframes(n_frames)
                                wav_arr = np.frombuffer(audio_data, dtype=np.int16)
                                wav_arr = wav_arr / 32768.0
                                if n_channels > 1:
                                    wav_arr = wav_arr.reshape(-1, n_channels).mean(axis=1)
                                waveform = torch.from_numpy(wav_arr).float().unsqueeze(0)
                        except:
                            # 文件损坏 → 跳过这个文件！
                            print(f"跳过损坏文件: {file_path}")
                            continue
                        
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                            
                        mfcc = mfcc_transform(waveform)
                        mfcc = mfcc.squeeze(0).transpose(0, 1) 
                        
                        seq_len = mfcc.shape[0]
                        if seq_len > self.max_len:
                            mfcc = mfcc[:self.max_len, :]
                        elif seq_len < self.max_len:
                            pad_len = self.max_len - seq_len
                            mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, pad_len))
                            
                        self.data_cache.append(mfcc)
                        self.labels.append(label)
        print(f"✅ 音频加载完成！共缓存 {len(self.data_cache)} 条音频特征。")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx], self.labels[idx]

# 音频加载器
def get_audio_dataloaders():
    full_audio_dataset = GTZANAudioDataset(root_dir=Config.AUDIO_DATA_DIR, max_len=Config.AUDIO_MAX_LEN)
    
    total = len(full_audio_dataset)
    train_sz = int(0.7 * total)
    val_sz = int(0.2 * total)
    test_sz = total - train_sz - val_sz # [cite: 30]
    
    gen = torch.Generator().manual_seed(Config.SEED)
    train_ds, val_ds, test_ds = random_split(
        full_audio_dataset, [train_sz, val_sz, test_sz], generator=gen
    )
    
    loader_kwargs = {'batch_size': Config.BATCH_SIZE, 'num_workers': 12, 'pin_memory': True}
    return (DataLoader(train_ds, shuffle=True, **loader_kwargs),
            DataLoader(val_ds, shuffle=False, **loader_kwargs),
            DataLoader(test_ds, shuffle=False, **loader_kwargs),
            train_ds)