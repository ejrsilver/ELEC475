
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer


def main():
    print('running main ...')
    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str,
                           help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size',
                           type=int, help='int [32]')
    args = argParser.parse_args()
    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = MNIST('./data/mnist', train=True,
                      download=True, transform=train_transform)
    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(
        N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0

    lin_int = torch.nn.Linear(8, 8)
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= train_set.data.size()[0]:
            img = train_set.data[idx]
            img = img.type(torch.float32)
            img = (img - torch.min(img)) / torch.max(img)
            img = img.to(device=device)
            img = img.view(1, img.shape[0]*img.shape[1]
                           ).type(torch.FloatTensor)
            noisy = img + torch.rand(784)
            with torch.no_grad():
                output = model(noisy)
            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output),
                  torch.min(output), torch.mean(output))
            N = 10
            f = plt.figure()
            with torch.no_grad():
                temp = model.encode(img)

            img = img.view(28, 28).type(torch.FloatTensor)
            noisy = noisy.view(28, 28).type(torch.FloatTensor)
            f.add_subplot(1, N+1, 1)
            plt.imshow(img, cmap='gray')

            for i in range(2, N+1):
                print(temp)
                with torch.no_grad():
                    temp = lin_int(temp)
                    imgtemp = (model.decode(temp)).view(
                        28, 28).type(torch.FloatTensor)
                    f.add_subplot(1, N+1, i)
                    plt.imshow(imgtemp, cmap='gray')

            # f = plt.figure()
            # f.add_subplot(1, 3, 1)
            # plt.imshow(img, cmap='gray')
            # f.add_subplot(1, 3, 2)
            # plt.imshow(noisy, cmap='gray')
            # f.add_subplot(1, 3, 3)
            # plt.imshow(output, cmap='gray')
            plt.show()


###################################################################
if __name__ == '__main__':
    main()
