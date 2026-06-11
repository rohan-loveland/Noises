import sys
print('Running in env:', sys.prefix)
try:
    import torch
    print('torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    import torchaudio
    print('torchaudio imported successfully')
    print('Environment is ready for spectrogram generation!')
except ImportError as e:
    print('Import error:', str(e))
    print('\nTorch is not installed in the Noises environment.')
    print('To install (recommended for CUDA 12.4):')
    print('conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia')
except Exception as e:
    print('Other error:', str(e))
