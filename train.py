import torch
import argparse
from tqdm import tqdm
from torch.nn import functional as F
from model import DNet
from torch.utils.tensorboard import SummaryWriter
from preprocess import DiagramDataset
from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='sketch to diagram')
    arg = parser.add_argument
    arg('--epochs',   type=int, default=7000, help='number of iterations')
    arg('--batch_size', type=int, default=16, help='number of iterations')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--save', type=str, default='checkpoint/', help='dir to save models')
    arg('--resume', type=str, default='', help='load saved model')
    arg('--log_interval', type=int, default=1, help='interval for log')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    for b_idx, (x, yt) in tqdm(enumerate(data_loader)):
        x = x.to(args.device)
        yt = yt.to(args.device)
        _, _, _, _, y = model(x)
        loss = criterion(y, yt)
        total_loss += loss.item()

    return total_loss * args.batch_size / len(data_loader.dataset)


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    total_loss = 0
    for b_idx, (x, yt) in tqdm(enumerate(data_loader)):
        x = x.to(args.device)
        yt = yt.to(args.device)
        y4, y3, y2, y1, y = model(x)
        # loss
        loss = criterion(y, yt)
        loss1 = criterion(y1, F.max_pool2d(yt, kernel_size=2, stride=2, padding=0))
        loss2 = criterion(y2, F.max_pool2d(yt, kernel_size=2, stride=2, padding=0))
        loss3 = criterion(y3, F.max_pool2d(yt, kernel_size=4, stride=4, padding=0))
        loss4 = criterion(y4, F.max_pool2d(yt, kernel_size=8, stride=8, padding=0))
        avg_loss = (loss+ (0.9*loss1) + (0.8*loss2) + (0.7*loss3) + (0.6*loss4))/5
        total_loss += avg_loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

    return total_loss * args.batch_size/len(data_loader.dataset)

def main(args):
    writer = SummaryWriter()

    train_dataset = DiagramDataset(root='./didi_dataset/',
                                   percent=0.8,
                                   transform=None)

    test_dataset = DiagramDataset(root='./didi_dataset/',
                                  percent=0.2,
                                  transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    model = DNet(n_classes=1)
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.BCELoss()

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optim_state'])
        print('Loaded model from {} checkpoint epoch # ({})'.format(args.resume, epoch))

    epoch = 0
    best = float('inf')
    while epoch <= args.epochs:
        train_error = train(model, train_dataloader, optimizer, criterion, args)
        test_error = test(model, test_dataloader, criterion, args)
        if epoch % args.log_interval == 0:
            print('{} Training Error: {} | Testing Error: {} '.format(epoch, train_error, test_error))
            writer.add_scalar('train error/epoch', train_error, epoch)
            writer.add_scalar('test error/epoch', test_error, epoch)
            # Save weights and model definition
            if test_error < best:
                best = test_error
                torch.save({
                    'epoch': epoch,
                    'model_def': model,
                    'state_dict': model.module.state_dict(),
                    'test_error': test_error,
                    'optim_state': optimizer.state_dict()}, f'{args.save}model_best.pth')

        epoch += 1

if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    print('Arguments:', args)
    main(args)
