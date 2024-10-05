import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import argparse
from torch.utils.data import DataLoader
from torchsummary import summary
from model import SnoutNet
from snoutnet_dataset import SnoutNetDataset

def flip_coordinates(coord):
    x, y = coord.tolist()

    #get the flipped version of x
    if x > 113:
        diff = x - 113
        new_x = 113 - diff
    else:
        diff = 113 - x
        new_x = 113 + x
    
    return torch.tensor([new_x, y])

#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 30
batch_size = 256
plot_file = 'plot.png'
device = 'cpu'
augmentation = 'just_data'


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        

def train(n_epochs, optimizer, model, train_loader, loss_fn, scheduler, device, save_file=None, plot_file=None):
    print("training...")
    model.train()
    losses_train = []

    for epoch in range(n_epochs):
        print("epoch: ", epoch)
        loss_train = 0.0
        for data in train_loader:
            imgs = data[0]
            labels = data[1]

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            loss = loss_fn(outputs.float(), labels.float())
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


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str,
                           help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size',
                           type=int, help='int [32]')
    argParser.add_argument('-e', metavar='epochs',
                           type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size',
                           type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str,
                           help='output loss plot file (.png)')
    argParser.add_argument('-a', metavar='augmentation', type=str,
                           help='augmentation_type')

    args = argParser.parse_args()

    if args.s != None:
        save_file = args.s
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p
    if args.a != None:
        augmentation = args.a
    
    
    model = SnoutNet()
    model.to(device=device)
    model.apply(init_weights)

    train_transform = transforms.Compose([transforms.ToTensor()])

    if augmentation == 'just_data':
        train_dataset = SnoutNetDataset(imgs_dir='./oxford-iiit-pet-noses/images-original/images', annotations_file='./oxford-iiit-pet-noses/train_noses.txt', transform=train_transform)
    elif augmentation == 'aug1':
        #horizontal flip
        normal_dataset = SnoutNetDataset(imgs_dir='./oxford-iiit-pet-noses/images-original/images', annotations_file='./oxford-iiit-pet-noses/train_noses.txt', transform=train_transform)
        flip_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=1)])
        flip_dataset = SnoutNetDataset(imgs_dir='./oxford-iiit-pet-noses/images-original/images', annotations_file='./oxford-iiit-pet-noses/train_noses.txt', transform=flip_transform, target_transform=flip_coordinates)
        train_dataset = torch.utils.data.ConcatDataset([normal_dataset, flip_dataset])

    elif augmentation == 'aug2':
        pass

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        train_loader=train_loader,
        scheduler=scheduler,
        loss_fn = loss_fn,
        device=device,
        save_file=save_file,
        plot_file=plot_file)

if __name__ == "__main__":
    main()