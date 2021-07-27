import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.models
from torch import optim
from torch.optim import lr_scheduler

from data_loader import DataLoader

img_num = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),
    'koniq-10k': list(range(0, 10073)),
    'bid': list(range(0, 586)),
    'palm': list(range(0, 600)),
    'palm2': list(range(0, 2600))
}


def load_data(dataset, img_size):
    index = img_size
    random.shuffle(index)
    # 选择80%参考图像生成的数据做训练集，剩下20%数据做测试集
    train_index = index[0:int(Round(0.8 * len(index)))]
    test_index = index[int(Round(0.8 * len(index))):]
    print(len(test_index))
    print(len(train_index))

    train_loader = DataLoader(dataset=dataset,
                              index=train_index,
                              batch_size=args.batch_size,
                              status='train',
                              num_worker=args.num_worker).get_loader()

    test_loader = DataLoader(dataset=dataset,
                             index=test_index,
                             batch_size=1,
                             status='test',
                             num_worker=args.num_worker).get_loader()

    return train_loader, test_loader


def fit(model, args):
    SRCC_List = []
    PLCCs_List = []
    return 0, 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="live")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--round", type=int, default=5)

    args = parser.parse_args()

    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.to(device)

    resnet50.fc = torch.nn.Linear(1000, 1)

    optimizer = optim.Adam([{'params': resnet50.parameters(), 'initial_lr': 1e-3}], lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 2, last_epoch=10)

    criterion = torch.nn.MSELoss

    Round = args.round

    SRCC_list = []
    PLCC_list = []
    for i in range(Round):
        SRCC, PLCC = fit(resnet50, args)
        SRCC_list.append(SRCC)
        PLCC_list.append(PLCC)
    print(resnet50)
