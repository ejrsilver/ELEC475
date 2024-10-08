
###############################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch, math, statistics
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from snoutnet_dataset import SnoutNetDataset
from model import SnoutNet
from PIL import Image, ImageDraw
from torchvision import transforms
batch_size = 698

def main():
    # Parse command line arguments.
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str,
                           help='parameter file (.pth)')
    argParser.add_argument('-i', type=int, help='index to show test on')
    args = argParser.parse_args()
    save_file = None
    if args.s is not None:
        save_file = args.s

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
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

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
            euclidean_distance = math.sqrt((outputs[0][0] - labels[0][0])**2 + (outputs[0][1] - labels[0][1])**2)
            euclidean_distances.append(euclidean_distance)
        
        minimum = min(euclidean_distances)
        maximum = max(euclidean_distances)
        average = statistics.mean(euclidean_distances)
        std = torch.std(torch.tensor(euclidean_distances)).item()

        print("Min: ", minimum, " Max: ", maximum, " Mean: ", average, "Standard Deviation: ", std)

    #test on an image with a given index
    idx = None
    if args.i is not None:
        idx = args.i
        img, label = test_dataset.__getitem__(idx)
        
        #run through model to get prediction
        with torch.no_grad():
            prediction = model(img)[0]
            x_prediction, y_prediction = prediction.tolist()
        
        transform_PIL = transforms.ToPILImage()
        img = transform_PIL(img)
        x, y = label.tolist()
        
        out_img = ImageDraw.Draw(img)
        out_img.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill='green')
        out_img.ellipse([(x_prediction - 3, y_prediction - 3), (x_prediction + 3, y_prediction + 3)], fill='red')
        img.show()

        

###################################################################
if __name__ == '__main__':
    main()
