import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import torchaudio
from torchcodec import decoders
import time

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")

NUM_OF_EPOCHS = 25
MAX_LENGTH = 2000 # Approximately 45 seconds (librosa sr = 22050, melspectro hop distance = 512)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3) # This layer is not used in the original forward pass, consider its purpose.
        
        # Calculate the input size for the LSTM
        # Dummy input for size calculation
        dummy_input = torch.randn(1, 1, 128, MAX_LENGTH)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        
        # The output of the CNN needs to be reshaped for the RNN. 
        # The new shape will be (batch_size, sequence_length, feature_size).
        # We will treat the time dimension (the last dimension) as the sequence length.
        conv_output_height = dummy_output.shape[2]
        conv_output_width = dummy_output.shape[3]
        
        cnn_output_features = 32 * conv_output_height
        
        # Recurrent layer (LSTM)
        self.lstm = nn.LSTM(input_size=cnn_output_features, hidden_size=128, batch_first=True)
        
        # Fully connected layer for the final output
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape for the LSTM layer
        # The input to LSTM should be (batch_size, sequence_length, features)
        # Here, the sequence is along the width (time) dimension of the spectrogram
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels * height, width).permute(0, 2, 1)
        
        # Pass through the LSTM
        # The output of LSTM is (output, (h_n, c_n))
        # We only need the output from the last time step for classification
        lstm_output, (h_n, c_n) = self.lstm(x)
        
        # Use the hidden state from the last time step
        # h_n shape is (num_layers * num_directions, batch, hidden_size)
        # We take the last layer's hidden state.
        last_hidden_state = h_n[-1] 
        
        # Pass through the final fully connected layer
        x = self.fc(last_hidden_state)
        
        return x


model = Net().to(device)

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
        y, sr = decoders.AudioDecoder(song_path, num_channels=1)
        y = y.mean(dim=0)
        mel_spectro = torchaudio.transforms.MelSpectrogram(sr)
        audio_tensor = mel_spectro(y).unsqueeze(0)

        # mean = audio_tensor.mean()
        # std = audio_tensor.std()
        # audio_tensor = (audio_tensor - mean) / x(std + 1e-6)

        bpm_label = self.bpm_labels.iloc[idx, 1]
        bpm_tensor = torch.tensor(bpm_label).float()

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        if self.target_transform:
            bpm_tensor = self.target_transform(bpm_tensor)
        return audio_tensor, bpm_tensor
    
def evaluate_mae(outputs, labels):
    mae = torch.mean(torch.abs(outputs - labels))
    return mae.item()

# (from Gemini)
def pad_collate(batch):
    max_len = MAX_LENGTH
    audio_tensors = []
    bpm_tensors = []

    for audio, bpm in batch:
        trimmed_audio = []
        if audio.shape[2] < max_len:
            padding_needed = max_len - audio.shape[2]
            trimmed_audio = F.pad(audio, (0, padding_needed))

        elif audio.shape[2] > max_len:
            trimmed_audio = audio[:, :, :max_len]


        audio_tensors.append(trimmed_audio)
        bpm_tensors.append(bpm)

    audio_batch = torch.stack(audio_tensors)
    bpm_batch = torch.stack(bpm_tensors)

    return audio_batch, bpm_batch

training_data = BPMDataset('bpm_train_TEST.csv','harmonixset/src/mp3s')
test_data = BPMDataset('bpm_test_TEST.csv','harmonixset/src/mp3s')

BATCH_SZ = 4

train_dataloader = DataLoader(training_data, batch_size=BATCH_SZ, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True, collate_fn=pad_collate)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # started with lr=5e-5

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Compute prediction error
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(inputs)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    mae = evaluate_mae(pred, y)
    print(f"Test Error: \n    MAE: {mae:>8f}, Avg loss: {test_loss:>8f}")
    return mae

for i, epoch in enumerate(range(NUM_OF_EPOCHS)):  # loop over the dataset multiple times
    start_time = time.time()
    print(f"----Epoch {i+1}----")
    # Call the train function to train the model for one epoch
    train(train_dataloader, model, loss_fn, optimizer)
    
    # Call the test function to evaluate the model after each epoch
    error = test(test_dataloader, model, loss_fn)

    if error < 5.0:
        model_path = f"models/bpm_model_epoch_{i+1}{time.time():0f}({error:.1f}).pth"
        torch.save(model.state_dict(), model_path)
    print(f"Elapsed time{time.time() - start_time} \n")

print('Finished Training')