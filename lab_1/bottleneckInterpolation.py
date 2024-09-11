
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
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

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
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
  
    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    N = 10
    idx = int(input("Enter index 1: "))
    idy = int(input("Enter index 2: "))
    
    imgX = train_set.data[idx]

    imgX = imgX.type(torch.float32)
    imgX = (imgX - torch.min(imgX)) / torch.max(imgX)

    imgX = imgX.to(device=device)
    imgX = imgX.view(1, imgX.shape[0]*imgX.shape[1]).type(torch.FloatTensor)

    with torch.no_grad():
        bottleneckX = model.encode(imgX)

    imgY = train_set.data[idy]

    imgY = imgY.type(torch.float32)
    imgY = (imgY - torch.min(imgY)) / torch.max(imgY)

    imgY = imgY.to(device=device)
    imgY = imgY.view(1, imgY.shape[0]*imgY.shape[1]).type(torch.FloatTensor)

    with torch.no_grad():
        bottleneckY = model.encode(imgY)


    # f1 = plt.figure()
    # f1.add_subplot(1, 2, 1)
    imgX = imgX.view(28, 28).type(torch.FloatTensor)
    # plt.imshow(imgX, cmap='gray')
    # f1.add_subplot(1, 2, 2)
    imgY = imgY.view(28, 28).type(torch.FloatTensor)
    # plt.imshow(imgY, cmap='gray')
    # plt.show()


    print("X: {}\nY: {}\n".format(bottleneckX,bottleneckY))
    temp = torch.lerp(bottleneckX, bottleneckY, 0.1)
    f = plt.figure()
    f.add_subplot(1, N+2, 1)
    plt.imshow(imgX, cmap='gray')
    for i in range(2, N+2):
        print(temp)
        with torch.no_grad():
            temp = torch.lerp(temp, bottleneckY, 0.1)
            imgtemp = (model.decode(temp)).view(28, 28).type(torch.FloatTensor)
            f.add_subplot(1, N+2, i)
            plt.imshow(imgtemp, cmap='gray')

    f.add_subplot(1, N+2, N+2)
    plt.imshow(imgY, cmap='gray')
    
    plt.show()








###################################################################

if __name__ == '__main__':
    main()



