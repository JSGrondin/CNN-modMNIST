from data_extraction import getdata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# CNN3.6-> baseline 3.4, with different images


# importing training and test features (i.e. images). The getdata() function
# returns only the digit that occupies the largest space or bounding square.
# Each element is a 2D array of size image_size X image_size
X_train = getdata('train', threshold=235, image_size=28)
X_test = getdata('test', threshold=235, image_size=28)

# normalizing data
X_train = X_train/255
X_test = X_test/255

# importing training labels
y_train = pd.read_csv('train_labels.csv')['Category']
y_train = np.asarray(y_train)

# splitting X_train and y_train into training and validation sets
random_seed = 123
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.05,
                                                  random_state=random_seed)

# Converting train, val and test sets into tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_val)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# Combining X_train_t and y_train_t into TensorDatasets, equivalently for
# the validation set.
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

# setting hyperparameters
n_epochs = 8
batch_size_train = 64
batch_size_val = 1000
batch_size_test = 10000
learning_rate = 0.1
momentum = 0.5
log_interval = 10
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
#
# Creating DataLoaders for the datasets
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size_val,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test_t,
                                           batch_size=batch_size_test,
                                           shuffle=True)

# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_train, shuffle=True)
#
# val_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_test, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# example_data.shape
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig


# Building the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#import pdb; pdb.set_trace() (n for next, l for list, ll pour long list,
# c for continue)
# Initializing the network and the optimizer
network = Net()
optimizer = optim.Adam(network.parameters())

# Training the model
train_losses = []
train_counter = []
val_losses = []
val_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test():
    network.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = network(data)
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(val_counter, val_losses, color='red')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
#
continued_network = Net()
continued_optimizer = optim.Adam(network.parameters())

network_state_dict = torch.load('./results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('./results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(n_epochs+2,n_epochs+3):
    val_counter.append(i*len(train_loader.dataset))
    train(i)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(val_counter, val_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig



examples = enumerate(test_loader)
batch_idx, (example_data) = next(examples)

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = network(X_test_t)

predict = np.zeros(len(example_data))

for i in range(len(example_data)):
    predict[i] = output.data.max(1, keepdim=True)[1][i].item()

# Predicting categories on test set and saving results as csv, ready for Kaggle
#my_prediction = lr.predict(X_combined_test)
my_prediction = np.array(predict).astype(int)
Id = np.linspace(0,9999, 10000).astype(int)
my_solution = pd.DataFrame(my_prediction, Id, columns = ['Category'])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['Id'])