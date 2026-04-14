import os
import torch

class Config:
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_DATA_DIR = './data/images_original'
    AUDIO_DATA_DIR = './data/genres_original'
    SAVE_DIR = './experiment_results'
    BATCH_SIZE = 32
    IMG_SIZE = (180, 180)
    CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
               'jazz', 'metal', 'pop', 'reggae', 'rock']
    NUM_CLASSES = 10
    AUDIO_MAX_LEN = 1300

# 自动创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)