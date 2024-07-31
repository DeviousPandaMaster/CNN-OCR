from matplotlib import pyplot as plt
import numpy as np
import os
import json
from matplotlib import pyplot as plt

TrainPath = os.path.expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/CharTrain.npz')
TestPath = os.path.expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/CharTest.npz')
decoderJson = os.path.expanduser('~/Desktop/ML-Stuff/CharacterOCR-torch/TMNIST-Decoder.json')
train, test = np.load(TrainPath), np.load(TestPath)
features_train_reshaped, target_train_categorical = train.get('arr_0'), train.get('arr_1')
features_test_reshaped, target_test_categorical = test.get('arr_0'), test.get('arr_1')
num_to_label_map = json.load(open(decoderJson, 'r'))
channel, img_height, img_width = features_train_reshaped[0].shape
num_classes = target_train_categorical.shape[-1]


print('Labels: ',num_to_label_map)
print('Train: ',features_train_reshaped.shape, target_train_categorical.shape)
print('Test: ',features_test_reshaped.shape, target_test_categorical.shape)
print('Channel: ',channel, ' Height: ',img_height, ' Width: ',img_width)

def TestShow(index):
    plt.imshow(features_train_reshaped[index].reshape(img_height, img_width, channel))
    testTar = np.argmax(target_train_categorical[index]) #[0]
    testTar = num_to_label_map[testTar]
    plt.title(testTar)
    plt.show()
 
import torch
from torch import nn, Tensor
F = nn.functional
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

class TMNIST_Net(torch.nn.Module):
    def __init__(self):
        super(TMNIST_Net, self).__init__()
        self.c1 = torch.nn.Conv2d(channel, 32, (4,4), stride=(4,4), padding=1)
        self.c2 = torch.nn.Conv2d(32, 64, (4,4), stride=(2,2), padding=1)
        self.l1 = torch.nn.Linear(576, 576)
        self.l2 = torch.nn.Linear(576, num_classes)
    
    def __call__(self, x):
        x = self.c1(x)
        x = F.leaky_relu(x)
        x = self.c2(x)
        x = F.leaky_relu(x)
        x = x.view(x.shape[0],-1)
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        return(x)
    
EPOCHS = 10
batch_size = -1
Device = 'cpu'

model = TMNIST_Net()
model.compile()
summary(model,(1,28,28))

class NPDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def Train():
    model.to(Device)
    images_tensor = torch.tensor(features_train_reshaped, dtype=torch.float32, device=Device)
    labels_tensor = torch.tensor(target_train_categorical, dtype=torch.float32, device=Device)
    dl = DataLoader(NPDataset(images_tensor,labels_tensor), batch_size=batch_size, shuffle=True)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), 0.0001)
    lossfn = nn.CrossEntropyLoss()
    for i in range(EPOCHS):
        epochLoss = 0.0
        for img, lab in dl:
            optim.zero_grad()
            out = model(img)
            loss = lossfn(out, lab)
            loss.backward()
            optim.step()
            epochLoss = loss.item()
        print('Epoch: ',i,' Loss: ',epochLoss)
    model.to('cpu')

Train()

def Test():
    with torch.no_grad():
        # Convert to PyTorch tensors
        model.to(Device)
        images_tensor = torch.tensor(features_test_reshaped, dtype=torch.float32, device=Device)
        labels_tensor = torch.tensor(target_test_categorical, dtype=torch.float32, device=Device)

        correct = 0
        for epoch in trange(0, images_tensor.shape[0]):  # Assuming you want to run for only 1 epoch
                img = images_tensor[epoch].reshape(1,1,28,28)
                lab = labels_tensor[epoch].argmax()
                out = model(img).argmax()  # Unsqueeze to add batch dimension
                if out == lab:
                    correct += 1
    print('Total Correct: ', correct, ' Out of: ', images_tensor.shape[0])
    print('Percent Correct: ', correct / images_tensor.shape[0] * 100)
    return correct

correct = Test()

torch.save(model, f'TMNIST-{correct}.pt')
