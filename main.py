import os
import sys
from pickletools import optimize
import torchvision
import torch
import torchvision.transforms as tr
from model import SimpleCNN
from transformer import VisualTransformer
import tqdm


def run_model(device, model, batch_size=144, learning_rate=1e-3, num_epochs=10, \
              save_name='cnn_model_control.pth'):
    """
    Preprocesses CIFAR10 data, trains the given model, tests it, and saves
    the final model as a .pth file with the given save_name.
    Computes and returns useful metrics (training loss, testing loss, and testing accuracy)
    :param device: Device on which to train the model, often
     torch.device('cuda:insert_num') or torch.device('cpu')
    :param model: A PyTorch model
    :param batch_size: Number of images to compute gradients for before updating weights
     Default of 144
    :param learning_rate: Hyperparamter that sets the rate at which gradient descent is
     performed. Default of 1e-3
    :param num_epochs: Number of epochs to train the model for
    :param save_name: Name of the file that will have the model's trainable parameters
     after training is complete
    :return: (average training loss, average testing loss, average testing accuracy)
    """
    
    model.to(device)

    transform = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,\
         download=True, transform=transform)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, \
        batch_size=batch_size, shuffle=True, num_workers=2)

    testing_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, \
        download=True, transform=transform)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, \
        batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loss = train(model, training_dataloader, loss_func, optimizer, num_epochs, device)
    test_loss, accuracy = test(model, testing_dataloader, loss_func, device)
    
    torch.save(model, save_name)
    
    return train_loss, test_loss, accuracy


def train(model, dataloader, loss_func, optimizer, num_epochs, device):
    """
    Trains the given model on the training dataset.
    :param model: A PyTorch model
    :param dataloader: A dataloader that contains the training dataset
     in the format (image, target).
    :param loss_func: A callable loss function (e.g. torch.nn.CrossEntropyLoss())
    :param optimizer: A callable optimizer used to update the model's trainable parameters
    :param num_epochs: Number of epochs to train the model for
    :param device: Device on which to train the model, often
     torch.device('cuda:insert_num') or torch.device('cpu')
    :return: A scalar (average loss over all epochs).
    """
    total_loss = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0

        with tqdm.tqdm(dataloader, unit="batch") as tepoch:
            for X, Y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                X, Y = X.to(device), Y.to(device)
                output = model(X)
                optimizer.zero_grad()
                loss = loss_func(output, Y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X.shape[0]

            epoch_loss = epoch_loss / len(dataloader.dataset)
            print("Epoch {}: {}".format(epoch, epoch_loss))
            total_loss += epoch_loss
    
    return total_loss / (num_epochs)
        


def test(model, dataloader, loss_func, device):
    """
    Computes the average loss and average accuracy of the given model on the test set.
    :param model: A PyTorch model
    :param dataloader: A dataloader that contains the testing dataset
     in the format (image, target).
    :param loss_func: A callable loss function (e.g. torch.nn.CrossEntropyLoss())
    :param device: Device on which to evaluate the model, often
     torch.device('cuda:insert_num') or torch.device('cpu')
    :return: A tuple. (average loss, average accuracy)
    """
    
    epoch_loss_sum = 0
    epoch_correct_sum = 0

    model.eval()
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for X, Y in tepoch:
            tepoch.set_description(f"Test progress")
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = loss_func(output, Y)
            epoch_loss_sum += loss.item() * X.shape[0]
            epoch_correct_sum += correct_predict_num(output, Y)

        avg_loss = epoch_loss_sum / len(dataloader.dataset)
        avg_accuracy = epoch_correct_sum / len(dataloader.dataset)

    return avg_loss, avg_accuracy

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


def main():
    
    model_types = {"CNN" : SimpleCNN, "TRANSFORMER" : VisualTransformer}
    
    if len(sys.argv) != 2 or sys.argv[1] not in model_types.keys():
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [CNN/TRANSFORMER]")
        exit()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    
    model = model_types[sys.argv[1]]()
    
    train_loss, test_loss, test_acc = run_model(device, model)
    print("Training loss: {}".format(train_loss))
    print("Testing Loss: {}".format(test_loss))
    print("Testing Accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()