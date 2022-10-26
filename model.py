import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Used to find the first music line in the picture. Can be lower in resolution.
class Network1stLine(nn.Module):
    def __init__(self):
        super(Network1stLine, self).__init__()
        # Convolution layers; image is first scaled to 320×320 with grayscale
        self.convs = nn.Sequential(
            nn.Conv2d(1, 2, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 6, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 8, 5),
        )
        self.dense = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU()
        )
        # Output 1: Whether there is a music line
        self.binary = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Output 2: The start and end y-position of music line
        self.coord = nn.Linear(16, 2)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        binary = self.binary(x)
        coord = self.coord(x)
        return torch.cat((binary, coord), 1)


def get_model_1st_line():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return Network1stLine().to(device), device


def loss_model_1st_line(target: torch.Tensor, result: torch.Tensor):
    """
    The custom loss function for my model.
    The value is in the structure of n×3 array, with the first column being the
    likelihood that there is a musical line, and the next two column being the
    upper and lower boundary of music notes, normalized to be as if the side
    length of the section is 1.
    The first column is kept as it is to be the loss derived from the
    probability; then the difference of the next two column is multiplied,
    because when there is no music lines, these two columns do not matter.
    """
    target_1st = target[:, 0]
    target_coord = target[:, 1:]
    result_1st = result[:, 0]
    result_coord = result[:, 1:]

    loss_likelihood = torch.mean((target_1st - result_1st) ** 2)
    loss_coord = torch.mean(target_1st * (target_coord - result_coord) ** 2)
    return loss_likelihood + loss_coord
