import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
# A helpful function for testing a single song at the end
from pathlib import Path
from tqdm import tqdm

def get_spectrogram_length(song_path):
    y, sr = librosa.load(song_path, sr=None)
    mel_spectro = librosa.feature.melspectrogram(y=y, sr=sr)
    # The time dimension is the last dimension of the spectrogram
    return mel_spectro.shape[1]

def find_max_length(annotations_file, song_dir):
    bpm_labels = pd.read_csv(annotations_file)
    max_len = 0
    
    # Use tqdm for a progress bar if your dataset is large
    # from tqdm import tqdm
    # for idx in tqdm(range(len(bpm_labels)), desc="Finding max length"):
    
    for idx in range(len(bpm_labels)):
        song_path = os.path.join(song_dir, bpm_labels.iloc[idx, 0])
        current_len = get_spectrogram_length(song_path)
        if current_len > max_len:
            max_len = current_len
            
    return max_len

# Define your file paths
csv_file_path = 'bpm_data.csv'
audio_dir_path = 'harmonixset/src/mp3s'

# Calculate the actual maximum length
max_time_dimension = find_max_length(csv_file_path, audio_dir_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(f"CUDA is available: {torch.cuda.is_available()}")

# --- Your Dataset and DataLoader (as provided) ---

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
        bpm_tensor = torch.tensor(bpm_label, dtype=torch.float32)

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        if self.target_transform:
            bpm_tensor = self.target_transform(bpm_tensor)

        return audio_tensor, bpm_tensor
    
FIXED_LENGTH = 50000
def fixed_collate(batch):
    audio_tensors = []
    bpm_tensors = []
    for audio, bpm in batch:
        curr_len = audio.shape[2]
        if curr_len > FIXED_LENGTH:
            # Truncate the tensor if it's too long
            processed_audio = audio[:, :, :FIXED_LENGTH]
        else:
            # Pad the tensor with zeros if it's too short
            padding_needed = FIXED_LENGTH - curr_len
            processed_audio = F.pad(audio, (0, padding_needed))
        
        audio_tensors.append(processed_audio)
        bpm_tensors.append(bpm)
    audio_batch = torch.stack(audio_tensors)
    bpm_batch = torch.stack(bpm_tensors)
    return audio_batch, bpm_batch

# --- Updated Neural Network Model ---
class CNNBPM(nn.Module):
    def __init__(self, max_time_dimension):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Use the pre-calculated maximum length for the dummy tensor
        dummy_input = torch.randn(1, 1, 128, max_time_dimension)
        
        with torch.no_grad():
            # Get the output shape from the convolutional layers
            conv_output_shape = self.conv_stack(dummy_input).shape
            
        # The flattened size is the product of the output dimensions after convolutions
        flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_layers(x)
        return logits.squeeze(1)


# --- Training and Testing Functions ---

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    with tqdm(dataloader, desc="Training") as pbar:
        for audio_batch, bpm_batch in pbar:
            audio_batch, bpm_batch = audio_batch.to(device), bpm_batch.to(device)

            pred = model(audio_batch)
            loss = loss_fn(pred, bpm_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    print(f"Average training loss: {total_loss / len(dataloader):.4f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(dataloader, desc="Testing") as pbar:
            for audio_batch, bpm_batch in pbar:
                audio_batch, bpm_batch = audio_batch.to(device), bpm_batch.to(device)
                
                pred = model(audio_batch)
                loss = loss_fn(pred, bpm_batch)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Average testing loss: {avg_loss:.4f}")

# --- Main Execution Block ---

# Load data and set up data loaders
training_data = BPMDataset('bpm_data.csv', 'harmonixset/src/mp3s')
test_data = BPMDataset('bpm_data.csv', 'harmonixset/src/mp3s')

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, collate_fn=fixed_collate)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=fixed_collate)

# Initialize model, loss function, and optimizer
model = CNNBPM(max_time_dimension).to(device)
loss_fn = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"--- Epoch {t+1} ---")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# --- New Function for Testing a Single Song ---

def test_single_song(model, song_path, device):
    model.eval()
    with torch.no_grad():
        # Load and preprocess the song
        y, sr = librosa.load(song_path, sr=None)
        mel_spectro = librosa.feature.melspectrogram(y=y, sr=sr)
        audio_tensor = torch.from_numpy(mel_spectro).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Get the prediction
        predicted_bpm = model(audio_tensor)
        
        return predicted_bpm.item()

# Example usage for a single song
song_to_test = Path('harmonixset/src/mp3s/0001_12step.mp3') # Change this to your test song
predicted_bpm = test_single_song(model, song_to_test, device)
print(f"Predicted BPM for {song_to_test.name}: {predicted_bpm:.2f}")