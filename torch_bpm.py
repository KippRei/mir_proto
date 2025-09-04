import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import time

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

MAX_LENGTH = 3000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)

        
        # Calculate the size of the flattened tensor dynamically
        # Start with a dummy input tensor of the same shape as your padded input
        dummy_input = torch.randn(1, 1, 128, MAX_LENGTH)
        
        # Pass it through the convolutional and pooling layers
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        
        # Determine the size after flattening
        flattened_size = dummy_output.view(-1).shape[0]
        
        # Initialize the linear layers with the correct size
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        y, sr = librosa.load(song_path, sr=None)
        mel_spectro = librosa.feature.melspectrogram(y=y, sr=sr)
        audio_tensor = torch.from_numpy(mel_spectro).float().unsqueeze(0)

        mean = audio_tensor.mean()
        std = audio_tensor.std()
        audio_tensor = (audio_tensor - mean) / (std + 1e-6)

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

# TODO: Study this more (from Gemini)
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

training_data = BPMDataset('bpm_train_TEST.csv', 'harmonixset/src/mp3s')
test_data = BPMDataset('bpm_test_TEST.csv', 'harmonixset/src/mp3s')

BATCH_SZ = 2

train_dataloader = DataLoader(training_data, batch_size=BATCH_SZ, shuffle=True, collate_fn=pad_collate)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SZ, shuffle=True, collate_fn=pad_collate)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# label = train_labels[0]
# print(f"Label: {label}")

# label1 = train_labels[1]
# print(f"Label2: {label1}")

# print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) # A better optimizer with an adaptive learning rate

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
    print(f"Test Error: \n MAE: {mae:>8f}, Avg loss: {test_loss:>8f} \n")
    return mae

for i, epoch in enumerate(range(10)):  # loop over the dataset multiple times
    start_time = time.time()
    print(f"----Epoch {i+1}----")
    # Call the train function to train the model for one epoch
    train(train_dataloader, model, loss_fn, optimizer)
    
    # Call the test function to evaluate the model after each epoch
    error = test(test_dataloader, model, loss_fn)

    model_path = f"models/bpm_model_epoch_{i+1}{time.time():0f}({error:.1f}).pth"
    torch.save(model.state_dict(), model_path)
    print(f"Elapsed time{time.time() - start_time}")

print('Finished Training')