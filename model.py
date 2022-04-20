from torch import nn

class SimpleCNN(nn.Module):

    def __init__(self, input_channels = 3, class_num = 10) -> None:
        super().__init__()

        
