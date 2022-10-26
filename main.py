import torch
import input_data

from model import get_model_1st_line, loss_model_1st_line


if __name__ == '__main__':
    model, device = get_model_1st_line()
    # print(model)

    x1 = torch.rand(2, 1, 320, 320, device=device)
    y1 = model(x1)

    x2 = torch.rand(2, 1, 320, 320, device=device)
    y2 = model(x2)
    a = loss_model_1st_line(y1, y2)
    # print(y)
    # input_data.get_sample_list()
