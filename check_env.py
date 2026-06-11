import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
import timm
print('timm available')
import pandas as pd
print('Data shape check:')
df = pd.read_csv('5sSpectrograms_tensors/train_5s_spectrograms.csv')
print('Total samples:', len(df))
print('Classes:', df['class_name'].value_counts().to_dict() if 'class_name' in df.columns else 'N/A')
print('Sample paths:', df['spectrogram_npy_path'].head(3).tolist())
