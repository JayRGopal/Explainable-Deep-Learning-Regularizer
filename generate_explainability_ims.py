import timm
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
from torchvision import models
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

def setup_dataloader():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Resize(224)
   
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=40,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader

def setup_resnet(model_type_folder, device):
    model_resnet50 = timm.create_model('resnet50', pretrained=True)
    model_resnet50.fc.weight = nn.Parameter(torch.rand(10, 2048))
    model_resnet50.fc.bias = nn.Parameter(torch.rand(10))

    path = "/gpfs/data/tserre/jgopal/trained-models/{}/model_best.pth.tar".format(model_type_folder) # <------- CHANGE THIS

    checkpoint = torch.load(path)
    model_resnet50.load_state_dict(checkpoint['state_dict'])
    model_resnet50.to(device).eval()

    return model_resnet50

def setup_vgg(model_type_folder, device):
    model_vgg19 = timm.create_model('vgg19', pretrained=True)
    model_vgg19.head.fc.weight = nn.Parameter(torch.rand(10, 4096))
    model_vgg19.head.fc.bias = nn.Parameter(torch.rand(10))

    path = "/gpfs/data/tserre/jgopal/trained-models/{}/model_best.pth.tar".format(model_type_folder)

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model_vgg19.load_state_dict(checkpoint['state_dict'])
    model_vgg19.to(device).eval()

    return model_vgg19

def test(model, dataloader, device):
    """
    Computes the average accuracy of the given model on the test set.
    :param model: A PyTorch model
    :param dataloader: A dataloader that contains the testing dataset
     in the format (image, target).
    :param device: Device on which to evaluate the model, often
     torch.device('cuda:insert_num') or torch.device('cpu')
    :return: average accuracy
    """
    
    epoch_correct_sum = 0

    model.eval()
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for X, Y in tepoch:
            tepoch.set_description(f"Test progress")
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            prev_sum = epoch_correct_sum 
            epoch_correct_sum = prev_sum + correct_predict_num(output, Y)

        avg_accuracy = epoch_correct_sum / len(dataloader.dataset)

    return avg_accuracy

def correct_predict_num(logit, target):
    """
    Returns the number of correct predictions.
    :param logit: 2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of the model.
    :param target: 1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.
    :return: A python scalar. The number of correct predictions.
    """
    
    predictions = torch.argmax(logit, dim = 1)
    predictions = torch.where(predictions == target, 1, 0)
    return torch.sum(predictions).item()

def attribute_image_features(model, algorithm, input, labels, ind=0, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=labels[ind],
                                              **kwargs
                                             )
    
    return tensor_attributions


def make_feature_maps(model, classes, images, labels, model_name):
    #class_indices = [0, 1, 2, 3, 4, 6, 10, 11, 12, 13, 25, 36]
    class_indices = [0, 1, 3, 4, 6, 11, 12, 13, 25, 36]
    store_ims = images
    store_labels = labels
    for ind in class_indices:
        images = store_ims[ind]
        im_one = images.cuda()
        images = images[None, :].cuda()
        labels = store_labels[ind].cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        print('Correct Class:', classes[labels])
        print('Predicted:', classes[predicted], 
            ' Probability:', torch.max(F.softmax(outputs, 1)).item())

        #input = images.unsqueeze(0)
        input = images
        input.requires_grad = True
        original_image = np.transpose((im_one.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        
        ## SALIENCY MAPS ## 
        saliency = Saliency(model)
        grads = saliency.attribute(input, target=labels.item())
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        ## INTEGRATED GRADIENTS ## 
        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(model, ig, input, labels.unsqueeze(0), baselines=input * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print('Approximation delta: ', abs(delta))
        # ig = IntegratedGradients(model)
        # nt = NoiseTunnel(ig)
        # attr_ig_nt = attribute_image_features(model, nt, input, labels.unsqueeze(0), baselines=input * 0, nt_type='smoothgrad_sq',
        #                                         nt_samples=100, stdevs=0.2)
        # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        ## DEEPLIFT ## 
        # dl = DeepLift(model)
        # attr_dl = attribute_image_features(model, dl, input, labels.unsqueeze(0), baselines=input * 0)
        # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        ## VISUALIZATION & SAVING ##
        # _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image", cmap='viridis', use_pyplot=False)
        _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                    show_colorbar=True, title="Overlayed Gradient Magnitudes", cmap='viridis', use_pyplot=False)
        canvas = FigureCanvas(_[0])
        _[0].savefig("/gpfs/data/tserre/jgopal/explainability/{}/{}_gradient.png".format(model_name, classes[labels]), dpi=100)
        _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="absolute_value",
                                   show_colorbar=True, title="Overlayed Integrated Gradients", cmap='viridis', use_pyplot=False)
        canvas = FigureCanvas(_[0])
        _[0].savefig("/gpfs/data/tserre/jgopal/explainability/{}/{}_integrated.png".format(model_name, classes[labels]), dpi=100)
        # _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
        #                             outlier_perc=10, show_colorbar=True,
        #                             title="Overlayed Integrated Gradients \n with SmoothGrad Squared", cmap='viridis', use_pyplot=False)
        # canvas = FigureCanvas(_[0])
        # _[0].savefig("/gpfs/data/tserre/jgopal/explainability/{}/{}_smoothGrad.png".format(model_name, classes[labels]), dpi=100)
        # _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="absolute_value",show_colorbar=True,
        #                             title="Overlayed DeepLift", cmap='viridis', use_pyplot=False)
        # canvas = FigureCanvas(_[0])
        # _[0].savefig("/gpfs/data/tserre/jgopal/explainability/{}/{}_deeplift.png".format(model_name, classes[labels]), dpi=100)

def main():
    model_types = {"RESNET" : setup_resnet, "VGG" : setup_vgg}
    
    if len(sys.argv) != 3 or sys.argv[1] not in model_types.keys():
        print("USAGE: python main.py <Model Type> <Model Type Folder>")
        print("<Model Type>: [RESNET/VGG]")
        print("<Model Type Folder>: [resnet-control/resnet-l2/...]")
        exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_types[sys.argv[1]](sys.argv[2], device)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    trainloader, testloader = setup_dataloader()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    #images = images.cuda()
    #labels = labels.cuda()
    #outputs = model(images)
    #_, predicted = torch.max(outputs, 1)

    #model_acc = test(model, testloader, device)
    #print(model_acc)

    make_feature_maps(model, classes, images, labels, sys.argv[2])

if __name__ == "__main__":
    main()