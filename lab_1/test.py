
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

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
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= train_set.data.size()[0]:
            print('label = ', train_set.targets[idx].item())
            img = train_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.to(device=device)
            # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            with torch.no_grad():
                output = model(img)
            # output = output.view(28, 28).type(torch.ByteTensor)
            # output = output.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))
            # plt.imshow(output, cmap='gray')
            # plt.show()

            # both = np.hstack((img.view(28, 28).type(torch.FloatTensor),output))
            # plt.imshow(both, cmap='gray')
            # plt.show()

            img = img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            f.add_subplot(1,2,1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1,2,2)
            plt.imshow(output, cmap='gray')
            plt.show()





###################################################################

if __name__ == '__main__':
    main()



