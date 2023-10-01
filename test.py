import torch
import argparse
import cv2
from tqdm import tqdm
from torch.nn import functional as F
from model import DNet
from torch.utils.tensorboard import SummaryWriter
from preprocess import DiagramDataset
from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='sketch to diagram')
    arg = parser.add_argument
    arg('--input', type=str, default='', help='input image')
    arg('--resume', type=str, default='checkpoint/model_best.pth', help='load saved model')
    arg('--log_interval', type=int, default=10, help='interval for log')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

def main(args):
    test_dataset = DiagramDataset(root='./didi_dataset/',
                                  percent=0.2,
                                  transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)

    model = DNet(n_classes=1)
    model = model.to(args.device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        test_error = checkpoint['test_error']
        print('Loaded model from {} checkpoint epoch # ({}) test_error {}'.format(args.resume, epoch, test_error))

    x, yt = next(iter(test_dataloader))
    with torch.no_grad():
        x = x.to(args.device)
        yt = yt.to(args.device)
        _, _, _, _, y = model(x)
        loss = criterion(y, yt)
        print('loss {}'.format(loss.item()))

    xx = x.cpu().numpy().transpose(2, 3, 1, 0).squeeze(3) * 255
    yyt = yt.cpu().numpy().transpose(2, 3, 1, 0).squeeze(3) * 255
    yy = y.cpu().numpy().transpose(2, 3, 1, 0).squeeze(3) * 255

    print(xx.shape)
    print(yyt.shape)
    print(yy.shape)
    cv2.imwrite("fig_x.png", xx)
    cv2.imwrite("fig_yt.png", yyt)
    cv2.imwrite("fig_y.png", yy)


if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    print('Arguments:', args)
    main(args)