from pickletools import optimize
import torchvision
import torch
import torchvision.transforms as tr
from model import SimpleCNN
import tqdm


def run_cnn(device):
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 1

    transform = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    testing_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = SimpleCNN()
    model.to(device)
    
    optimizer = torch.optim.Adam(SimpleCNN.parameters(model), learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loss = train(model, training_dataloader, loss_func, optimizer, num_epochs, device)
    test_loss, accuracy = test(model, testing_dataloader, loss_func, optimizer, device)

    return train_loss, test_loss, accuracy

def train(model, dataloader, loss_func, optimizer, num_epochs, device):
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

            print("Epoch {}: {}".format(epoch, epoch_loss))
            total_loss += epoch_loss
    
    return total_loss / num_epochs
        


def test(model, dataloader, loss_func, optimizer, device):
    
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
        avg_correct = epoch_correct_sum / len(dataloader.dataset)

    return avg_loss, avg_correct

def correct_predict_num(logit, target):
    """
    Returns the number of correct predictions.
    :param logit: 2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of CNN model.
    :param target: 1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.
    :return: A python scalar. The number of correct predictions.
    """
    predictions = torch.argmax(logit, dim = 1)
    predictions = torch.where(predictions == target, 1, 0)
    return torch.sum(predictions).item()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loss, test_loss, test_acc = run_cnn(device)
    print("Training loss: {}".format(train_loss))
    print("Testing Loss: {}".format(test_loss))
    print("Testing Accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()