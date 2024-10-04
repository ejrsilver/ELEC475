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

#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 5
batch_size = 1
plot_file = 'plot.png'
device = 'cpu'


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
    model = SnoutNet()
    model.to(device=device)
    model.apply(init_weights)

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = train_transform

    train_dataset = SnoutNetDataset(imgs_dir='./oxford-iiit-pet-noses/images-original/images', annotations_file='./oxford-iiit-pet-noses/train_noses.txt', transform=train_transform)
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