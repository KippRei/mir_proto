import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import os

# Create a dummy dataset directory with some files for this example
# In a real scenario, you'd have your own audio files and a CSV/JSON file with labels.
# E.g., data/
#        ├── audio_1.wav (BPM: 120)
#        ├── audio_2.wav (BPM: 90)
#        └── ...
# For this example, we'll create a simple list of paths and labels.
audio_paths = 'harmonixset/src/mp3s'
bpm_labels = 'bpm_train_TEST.csv'

class BPMDataset(Dataset):
    def __init__(self, audio_paths, bpm_labels, sr=22050, n_mels=128):
        self.audio_paths = audio_paths
        self.bpm_labels = bpm_labels
        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        bpm = self.bpm_labels[idx]

        # Load audio and resample if necessary
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Feature Extraction: Mel Spectrogram is a common choice for audio tasks.
        # It represents the frequency content of the audio over time.
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_mels=self.n_mels
        )
        mel_spec = mel_spectrogram_transform(waveform)

        # You might also want to compute other features, like the onset detection function,
        # which is a strong indicator of beat locations.
        onset_env = librosa.onset.onset_strength(y=waveform.squeeze().numpy(), sr=self.sr)
        
        # In a real-world model, you'd stack these features or use them to build a more
        # complex input representation. For this simplified example, we'll just use the Mel spectrogram.
        
        # Truncate or pad the mel spectrogram to a fixed size for batching
        max_length = 512
        if mel_spec.shape[2] > max_length:
            mel_spec = mel_spec[:, :, :max_length]
        else:
            padding = torch.zeros(1, self.n_mels, max_length - mel_spec.shape[2])
            mel_spec = torch.cat((mel_spec, padding), dim=2)

        return mel_spec, torch.tensor(bpm, dtype=torch.float32)

# Instantiate the dataset and data loader
# Replace dummy data with your actual file paths and labels
dataset = BPMDataset(audio_paths, bpm_labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class BPMDetectorCNN(nn.Module):
    def __init__(self, n_mels, output_size=1):
        super(BPMDetectorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Calculate the size of the flattened tensor after conv layers
        # This is a bit tricky, but you can compute it based on your input size and pooling layers.
        # A simpler way is to run a dummy tensor through the network to get the size.
        # For our example, let's assume a flattened size.
        dummy_input = torch.randn(1, 1, n_mels, 512)
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def _get_flattened_size(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

# Instantiate the model
n_mels = 128
model = BPMDetectorCNN(n_mels)

import torch.optim as optim

# Define hyperparameters
epochs = 10
learning_rate = 0.001

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # The model's output is a single value, so we squeeze the labels to match
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Finished Training')

# You would save your trained model for future use
torch.save(model.state_dict(), 'bpm_detector.pth')