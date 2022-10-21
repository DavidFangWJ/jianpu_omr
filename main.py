import torch
import input_data

from model import get_model_1st_line


if __name__ == '__main__':
    model, device = get_model_1st_line()
    print(model)

    x = torch.rand(2, 1, 320, 320, device=device)
    y = model(x)
    print(y)
    # input_data.get_sample_list()
