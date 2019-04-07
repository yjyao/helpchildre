import random
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
from sys import argv


def progress_bar(progress, size=10, arrow='>'):
    pos = int(min(progress, 1.0) * size)
    return '[{bar:<{size}.{trim}}]'.format(
        bar='{:=>{pos}}'.format(arrow, pos=pos),
        size=size-1,  # Then final tick is covered by the end of bar.
        trim=min(pos,size-1))

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.01              # learning rate
DATA_DIR = './data/'
DOWNLOAD_CIFAR10 = not(os.path.exists(DATA_DIR)) or not os.listdir(DATA_DIR)
seed = random.randint(1, 10**6)

if len(argv) > 1:
    EPOCH = int(argv[1])
if len(argv) > 2:
    BATCH_SIZE = int(argv[2])
if len(argv) > 3:
    LR = float(argv[3])
if len(argv) > 4:
    seed = int(argv[4])

print('Using seed {}'.format(seed))
np.random.seed(seed)
torch.manual_seed(seed)

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

    class ConvLayer(nn.Sequential):
        '''Conv2D layer with activation function and optionally MaxPool.

        NOTE Padding is by default set to (kernel_size - 1) // 2 instead of 0.
        '''

        def __init__(self, in_size,
                     in_channels, out_channels, kernel_size,
                     stride=1, padding=None, dilation=1, groups=1, bias=True,
                     activation=None,
                     max_pool_size=1,
                     dropout=0):
            padding = (kernel_size - 1) // 2 if padding is None else padding
            super(CNN.ConvLayer, self).__init__(*[m for m in (
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, bias),
                activation,
                nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
                if max_pool_size > 1 else None,
                nn.Dropout(dropout) if dropout else None,
            ) if m])
            self.out_channels = out_channels
            self.out_size = ((in_size +
                              2 * padding -
                              dilation * (kernel_size - 1) - 1) //
                             stride + 1) // (max_pool_size or 1)
            self.out_features = self.out_channels * self.out_size**2

    class FcLayer(nn.Sequential):

        def __init__(self, in_features, out_features, bias=True,
                     activation=None,
                     dropout=0,
                     batchnorm=False):
            super(CNN.FcLayer, self).__init__(*[m for m in (
                nn.BatchNorm1d(in_features) if batchnorm else None,
                nn.Linear(in_features, out_features, bias),
                activation,
                nn.Dropout(dropout) if dropout else None,
            ) if m])
            self.out_features = out_features

    def __init__(self):
        super(CNN, self).__init__()

        self.IMAGE_SIZE = 32
        self.IMAGE_CHANNELS = 3
        self.NUM_CLASSES = 10

        self.conv_layers = []
        self.fc_layers = []

        self.add_conv_layer(
            out_channels=16,
            kernel_size=5,
            activation=nn.ReLU(),
            dropout=0.25,
        )
        self.add_conv_layer(
            out_channels=16,
            kernel_size=3,
            activation=nn.ReLU(),
            max_pool_size=2,
            dropout=0.25,
        )
        self.add_conv_layer(
            out_channels=16,
            kernel_size=5,
            activation=nn.ReLU(),
            dropout=0.5,
        )
        self.add_conv_layer(
            out_channels=16,
            kernel_size=3,
            stride=2,
            activation=nn.ReLU(),
            dropout=0.5,
        )

        self.add_fc_layer(
            batchnorm=True,
            out_features=218,
            activation=nn.ReLU(),
        )
        self.add_fc_layer(out_features=self.NUM_CLASSES)

        self.encoder = nn.Sequential(*self.conv_layers)
        self.decoder = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

    def add_conv_layer(self, **kw_args):
        if not self.conv_layers:
            in_size = self.IMAGE_SIZE
            in_channels = self.IMAGE_CHANNELS
        else:
            in_size = self.conv_layers[-1].out_size
            in_channels = self.conv_layers[-1].out_channels
        self.conv_layers.append(CNN.ConvLayer(
            in_size=in_size, in_channels=in_channels, **kw_args))

    def add_fc_layer(self, **kw_args):
        if not self.fc_layers:
            in_features = self.conv_layers[-1].out_features
        else:
            in_features = self.fc_layers[-1].out_features
        self.fc_layers.append(CNN.FcLayer(in_features=in_features, **kw_args))

cnn = CNN()
print(cnn)  # net architecture
print('Using {:,} parameters'.format(sum(p.numel() for p in cnn.parameters())))
print([(n, p.numel()) for n, p in cnn.named_parameters()])

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

def validate(loader, name, old_loss=None, old_acc=None):
    cnn.eval()  # eval mode (different batchnorm, dropout, etc.)
    with torch.no_grad():
        correct = 0
        loss = 0
        for images, labels in loader:
            images, labels = Variable(images), Variable(labels)
            outputs = cnn(images)
            _, predicts = torch.max(outputs.data, 1)
            correct += (predicts == labels).sum().item()
            loss += loss_func(outputs, labels).item()
    sign = lambda x: x and (-1, 1)[x>0]
    compsymb = lambda v: {-1: 'v', 0: '=', 1: '^'}[sign(v)]
    avg_loss, acc = loss / len(loader), correct / len(loader.dataset)
    print(('[{name} images]'
           '\t avg loss: {avg_loss:5.3f}{loss_comp}'
           ', accuracy: {acc:6.2f}%{acc_comp}').format(
               name=name, avg_loss=avg_loss, acc=100 * acc,
               loss_comp='' if old_loss is None else compsymb(avg_loss-old_loss),
               acc_comp='' if old_acc is None else compsymb(acc-old_acc)))
    return avg_loss, acc

# training and testing
print('TRAINING')
print('='*30)
print('''\
Epoches: {EPOCH}
Batch size: {BATCH_SIZE}
Learning rate: {LR}
'''.format(**locals()))
running_loss_size = max(1, len(train_loader) // 10)
train_loss, train_accuracy = None, None
test_loss, test_accuracy = None, None
for epoch in range(EPOCH):
    running_loss = 0.0
    cnn.train()  # train mode
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
            print('[{}] Epoch {} {} loss: {:.3f}'.format(
                datetime.datetime.now().strftime('%H:%M:%S'),
                epoch + 1,
                progress_bar((i+1) / len(train_loader)),
                running_loss / running_loss_size))
            running_loss = 0.0
    train_loss, train_accuracy = validate(train_loader, 'train', train_loss, train_accuracy)
    test_loss, test_accuracy = validate(test_loader, 'test', test_loss, test_accuracy)
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

print('Results generated with seed {}'.format(seed))
