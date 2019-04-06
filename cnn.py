import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import numpy as np
import datetime
from torch.autograd import Variable
import torch.nn.functional as F


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 200               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
LR = 0.01              # learning rate
DATA_DIR = './data/'
DOWNLOAD_CIFAR10 = not(os.path.exists(DATA_DIR)) or not os.listdir(DATA_DIR)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,                                     # this is training data
    transform=transform,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_CIFAR10
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


test_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    transform=transform,
)

test_loader = Data.DataLoader(dataset=test_data, batch_size=(BATCH_SIZE//2), shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#=====================================#=====================================#
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 *16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# # following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()


# training and testing
for epoch in range(EPOCH):
    running_loss = 0.0
    running_loss_size = len(train_loader) // 10
    for i, data in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # clear gradients for this training step
        optimizer.zero_grad()   
        # forward + backward + optimize
        outputs = cnn(inputs)               # cnn output
        loss = loss_func(outputs, labels)   # cross entropy loss
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # print statistics
        running_loss += loss.item()
        if i % running_loss_size == running_loss_size - 1:
            print('[%d, %5d][%s] loss: %.3f' %
                  (epoch + 1, i + 1,
                   datetime.datetime.now().strftime('%H:%M:%S'),
                   running_loss / running_loss_size))
            running_loss = 0.0
print('Finished Training')

#         if step % 50 == 0:
#             test_output, last_layer = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#             if HAS_SK:
#                 # Visualization of trained flatten layer (T-SNE)
#                 tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                 plot_only = 500
#                 low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
#                 labels = test_y.numpy()[:plot_only]
#                 plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# # print 10 predictions from test data
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
