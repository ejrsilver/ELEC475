
###############################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch, math
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from snoutnet_dataset import SnoutNetDataset
from model import SnoutNet
batch_size = 1


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
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = SnoutNetDataset(imgs_dir='./oxford-iiit-pet-noses/images-original/images', annotations_file='./oxford-iiit-pet-noses/test_noses.txt', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
   

    model = SnoutNet()

    if save_file is not None:
        model.load_state_dict(torch.load(save_file))

    model.to(device)
    model.eval()

    #to hold euclidean distances for analysis
    euclidean_distances = []
    with torch.no_grad():
        for data in test_loader:
            imgs = data[0]
            labels = data[1]

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            for idx in range(len(labels)):
                euclidean_distance = math.sqrt((outputs[idx][0] - labels[idx][0])**2 + (outputs[idx][1] - labels[idx][1])**2)
                euclidean_distances.append(euclidean_distance)

        min = min(euclidean_distances)
        max = max(euclidean_distances)
        mean = mean(euclidean_distances)
        std = torch.std(torch.tensor(euclidean_distances))

        print("Min: ", min, " Max: ", max, " Mean: ", mean, "Standard Deviation: ", std)
        




###################################################################
if __name__ == '__main__':
    main()
