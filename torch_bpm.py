import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

MAX_LENGTH = 3000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        
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

        bpm_label = self.bpm_labels.iloc[idx, 1]
        bpm_tensor = torch.tensor(bpm_label).float()

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        if self.target_transform:
            bpm_tensor = self.target_transform(bpm_tensor)

        return audio_tensor, bpm_tensor
    
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

training_data = BPMDataset('bpm_test_run.csv', 'harmonixset/src/mp3s')
test_data = BPMDataset('bpm_test_run.csv', 'harmonixset/src/mp3s')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

BATCH_SZ = 1

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for i, epoch in enumerate(range(2)):  # loop over the dataset multiple times
    print(f"----Epoch {i+1}----")
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # move data to gpu
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')