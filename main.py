import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def display_random_spectrogram(data_dir: str, n_mels: int = 128, fmax: int = 16000):
    """
    Pick a random .ogg file from the BirdCLEF train_audio directory (recursive)
    and display its mel spectrogram.
    """
    # Find all .ogg files recursively
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.ogg'):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found in {data_dir}")

    # Pick random file
    file_path = random.choice(audio_files)
    print(f"Selected file: {file_path}")

    # Load audio
    y, sr = librosa.load(file_path, sr=None)  # keep original sample rate

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        fmax=fmax
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram\n{os.path.basename(file_path)}')
    plt.tight_layout()
    plt.show()


# ============== Usage ==============
if __name__ == "__main__":
    # <<< CHANGE THIS TO YOUR DIRECTORY >>>
    DATA_DIR = "Data/train_audio"  # or "/path/to/your/birdclef/train_audio"

    display_random_spectrogram(DATA_DIR)