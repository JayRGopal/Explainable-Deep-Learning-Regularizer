# Reg-Explain
Evaluating the impact of regularization on explainability

A custom-built simple CNN has been coded via PyTorch to aid in this endeavor.

Additionally, Resnet50 is being used. It is being trained via code from the PyTorch-Image-Models (TIMM) repository.

The dataset being used is CIFAR10.


# Notes
Folders labeled as "OLD" have an old method of formatting that is no longer favored. They remain on the GitHub for reference.

The custom-built transformer has not undergone a thorough explainability analysis yet.

# Resnet50 310 Epoch Results
Resnet50-Control: 96.48%

Resnet50-Dropout: 96.34%

Resnet50-L2: 96.42%


# Vgg19 310 Epoch Results
Vgg19-Control: 91.45%

Vgg19-Dropout: 91.83%

Vgg19-L2: 91.37%


# SimpleCNN 8 Epoch Results
SimpleCNN-Control: 64.00%

SimpleCNN-L2 1e-5: 66.63%

SimpleCNN-L2 1e-8: 65.83%

SimpleCNN-Dropout: 68.32%

SimpleCNN-Dropout-L2 1e-5: 68.06%

SimpleCNN-Dropout-L2 1e-8: 68.76%

