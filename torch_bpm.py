import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class BPMDataset(Dataset):
    def __init__(self, annotations_file, song_dir, transform=None, target_transform=None):
        self.bpm_labels = pd.read_csv(annotations_file)
        self.song_dir = song_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bpm_labels)
    
    def __getitem__(self, idx):
        song_path = os.path.join(self.song_dir, self.bpm_labels.iloc[idx, 0])
        y, sr = librosa.load(song_path, sr=None)
        mel_spectro = librosa.feature.melspectrogram(y=y, sr=sr)
        audio_tensor = torch.from_numpy(mel_spectro).float().unsqueeze(0)

        bpm_label = self.bpm_labels.iloc[idx, 1]
        bpm_tensor = torch.tensor(bpm_label)

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        if self.target_transform:
            bpm_tensor = self.target_transform(bpm_tensor)

        return audio_tensor, bpm_tensor
    
# TODO: Study this more (from Gemini)
def pad_collate(batch):
    max_len = max(item[0].shape[2] for item in batch)
    audio_tensors = []
    bpm_tensors = []
    for audio, bpm in batch:
        padding_needed = max_len - audio.shape[2]
        padded_audio = F.pad(audio, (0, padding_needed))

        audio_tensors.append(padded_audio)
        bpm_tensors.append(bpm)

    audio_batch = torch.stack(audio_tensors)
    bpm_batch = torch.stack(bpm_tensors)

    return audio_batch, bpm_batch

training_data = BPMDataset('bpm_data.csv', 'harmonixset/src/mp3s')
test_data = BPMDataset('bpm_data.csv', 'harmonixset/src/mp3s')

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=pad_collate)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

label = train_labels[0]
print(f"Label: {label}")

model = NeuralNetwork().to(device)
print(model)
