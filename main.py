import torch
import numpy as np
import math


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()
