from tkinter import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch import LongTensor
from PIL import Image
import numpy as np
import os

class CIFAR10_Dataset(Dataset):

    def __init__(self, X, Y, height = 32, width = 32, channels = 3) -> None:
        """
        Initializes CIFAR10 dataset

        :param X: Features of [n, 1024] where n is sample size
        :param Y: Labels of shape [n,] where n is sample size
        :param height: height of each image
        :param width: width of each image
        :param channels: number of channels in each image
        """
        super().__init__()

        self.X = X
        self.Y = LongTensor(Y)

        self.h = height
        self.w = width
        self.c = channels

    def __len__(self):
        """
        Returns length of dataset
        """

        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Reshapes a sample feature of shape [1024] to [self.c, self.h, self.w].
        Returns the reshaped feature and label of the sample at the given index.
        :param index: Index of a sample.
        :return: Reshaped feature and label of the sample at the given index.
        """
        x = self.X[index]
        x = np.reshape(x, (self.c, self.h, self.w))
        return x, self.Y[index]

# def get_cifar_loader(batch_size, dataset="data/cifar10", test_size=0.2,
#                      shuffle_train_label=False, shuffle_ratio=0.8):
#     """
#     Returns train/test dataloaders of MNIST dataset.
#     :param batch_size: Batch size.
#     :param dataset: Path to MNIST dataset digits.csv.
#     :param test_size: Ratio of (test set size / dataset size)
#     :param shuffle_train_label: If True, shuffle the training labels.
#         Useful for the third report question.
#     :param shuffle_ratio: Proportion of training labels to shuffle.
#     :return: Dataloaders for training set and test set.
#     """
#     # Check if the file exists
#     if not os.path.exists(dataset):
#         raise FileNotFoundError('The file {} does not exist'.format(dataset))

#     loader = transforms.Compose([transforms.Scale(256), transforms.ToTensor()])
#     train_dataset = datasets.ImageFolder("")

#     # for label in os.scandir("cifar10/train"):
#     #     for image_path in os.scandir(label):
#     #         imsize = 256
#     #         
#     #         image = Image.open(image_path)
#     #         image = loader(image).float()
            



#     # We assume labels are in the first column of the dataset
#     Y = data.values[:, 0].astype(np.int32)

#     # Features columns are indexed from 1 to the end, make sure that dtype = float32
#     X = data.values[:, 1:].astype(np.float32)

#     # Split data into training set and test set
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

#     # Build dataset
#     dataset_train = CIFAR10_Dataset(X_train, Y_train)
#     dataset_test = CIFAR10_Dataset(X_test, Y_test)

#     # Build dataloader
#     dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
#     dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

#     return dataloader_train, dataloader_test