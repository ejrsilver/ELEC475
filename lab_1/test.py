
###############################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer


def main():
    # Parse command line arguments.
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str,
                           help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size',
                           type=int, help='int [32]')
    args = argParser.parse_args()
    save_file = None
    if args.s is not None:
        save_file = args.s
    bottleneck_size = 0
    if args.z is not None:
        bottleneck_size = args.z

    # Select processing device (modified to support Vulkan).
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.is_vulkan_available():
        device = 'vulkan'
    print('\t\tusing device ', device)
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = MNIST('./data/mnist', train=True,
                      download=True, transform=train_transform)
    N_input = 784
    N_output = N_input

    model = autoencoderMLP4Layer(
        N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)

    if save_file is not None:
        model.load_state_dict(torch.load(save_file))

    model.to(device)
    model.eval()

    idx = 0
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
                output = model(img)
                output2 = model(noisy)

            img = img.view(28, 28).type(torch.FloatTensor)
            noisy = noisy.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            output2 = output2.view(28, 28).type(torch.FloatTensor)

            print("Step 2.")
            plt.imshow(train_set.data[idx], cmap="gray")
            plt.show()

            print("Step 4.")
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.show()

            print("Step 5.")
            f = plt.figure()
            f.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 3, 2)
            plt.imshow(noisy, cmap='gray')
            f.add_subplot(1, 3, 3)
            plt.imshow(output2, cmap='gray')
            plt.show()

            print("Step 6.")
            jdx = input("Enter another index > ")
            jdx = int(jdx)
            if 0 <= jdx <= train_set.data.size()[0]:
                img = train_set.data[idx]
                img2 = train_set.data[jdx]

                N = 10
                f = plt.figure()
                f.add_subplot(1, N+1, 1)
                plt.imshow(img, cmap='gray')
                f.add_subplot(1, N+1, N+1)
                plt.imshow(img2, cmap='gray')

                img = img.type(torch.float32)
                img = (img - torch.min(img)) / torch.max(img)
                img = img.to(device=device)
                img = img.view(
                    1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)

                img2 = img2.type(torch.float32)
                img2 = (img2 - torch.min(img2)) / torch.max(img2)
                img2 = img2.to(device=device)
                img2 = img2.view(
                    1, img2.shape[0]*img2.shape[1]).type(torch.FloatTensor)

                with torch.no_grad():
                    img = model.encode(img)
                    img2 = model.encode(img2)
                    for i in range(2, N+1):
                        img = torch.lerp(img, img2, 0.2)
                        output = (model.decode(img)).view(
                            28, 28).type(torch.FloatTensor)
                        f.add_subplot(1, N+1, i)
                        plt.imshow(output, cmap='gray')
                    plt.show()


###################################################################
if __name__ == '__main__':
    main()
