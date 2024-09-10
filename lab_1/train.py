
#########################################################################################################
#
#   ELEC 475 - Lab 1
#   Fall 2023
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import argparse
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchsummary import summary
from model import autoencoderMLP4Layer

#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 30
batch_size = 256
bottleneck_size = 32
plot_file = 'plot.png'

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
    print('training ...')
    model.train()

    losses_train = []
    for epoch in range(1, n_epochs+1):

        print('epoch ', epoch)
        loss_train = 0.0
        for data in train_loader:
            imgs = data[0]
            imgs= imgs.view(imgs.shape[0], -1)  #   Q1/ What does this line do, and why is it needed?
            #   print('break 8 : ', imgs.shape, imgs.dtype)
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():

    global bottleneck_size, save_file, n_epochs, batch_size

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    args = argParser.parse_args()

    if args.s != None:
        save_file = args.s
    if args.z != None:
        bottleneck_size = args.z
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p

    print('\t\tbottleneck size = ', bottleneck_size)
    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    # test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
            save_file=save_file,
            plot_file = plot_file)

###################################################################

if __name__ == '__main__':
    main()



