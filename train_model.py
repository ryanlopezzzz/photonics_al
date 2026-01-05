import argparse
import logging
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.phc_dataset import PhC2DBandgapQuickLoad
from augmentations import get_aug
from model_architecture import RegressionAnalyticBNN

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)

def create_and_train_model(args, train_loader, test_loader, save=True):
    model = RegressionAnalyticBNN()
    model.to(args.device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+1)
    train_model(args, train_loader, test_loader, model, optimizer, scheduler, save=save)
    return model

def train_model(args, train_loader, test_loader, model, optimizer, scheduler, save=True):
    for epoch in range(1, args.epochs+1):
        scheduler.step()
        train_start_time = time.time()
        loss, loss_prediction, loss_variance, loss_kl = train_epoch(args, train_loader, model, optimizer)
        train_end_time = time.time()
        if epoch == args.epochs:
            test_error = test(args, test_loader, model)
        else:
            test_error = -1.0
        test_end_time = time.time()

        logger.info(f'Epoch: {epoch:03d}, LR: {scheduler.get_last_lr()[0]:7f}, Loss: {loss:.7f}, Loss_pred: {loss_prediction:.7f}, Loss_variance: {loss_variance:.7f}'
            f' Test MAE: {test_error:.7f}, Train Time: {train_end_time-train_start_time:1f}, Test Time = {test_end_time-train_end_time}')
        if save == True:
            args.writer.add_scalar(f'train/train_loss', loss, epoch)
            args.writer.add_scalar(f'train/loss_prediction', loss_prediction, epoch)
            args.writer.add_scalar(f'train/extra_bnn_loss', loss_variance, epoch)
            args.writer.add_scalar(f'train/loss_kl', loss_kl, epoch)
            args.writer.add_scalar(f'test/test_error', test_error, epoch)

            if (epoch == args.epochs) or (args.save_every_n_epochs != None and epoch % args.save_every_n_epochs == 0):
                    torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),}, 
                        os.path.join(args.out, f'epoch_{epoch}_checkpoint.pth.tar'))

def train_epoch(args, train_loader, model, optimizer):
    model.train()
    loss_all = 0
    loss_predictions = 0
    loss_variances = 0
    loss_kls = 0

    for (inputs, targets) in train_loader:
        inputs = get_aug(inputs, translate=True,rotate=True,flip=True)
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()

        predictions, kl, output_variance = model(inputs)
        loss_prediction = F.mse_loss(predictions, targets)
        loss_kl = args.lambda_kl * kl
        loss_variance = torch.mean(output_variance)
        loss = loss_prediction + loss_kl + loss_variance

        loss.backward()
        optimizer.step()

        loss_all += loss.item() * len(inputs)
        loss_predictions += loss_prediction.item() * len(inputs)
        loss_variances += loss_variance.item() * len(inputs)
        loss_kls += loss_kl.item() * len(inputs)
    return loss_all / len(train_loader.dataset), loss_predictions / len(train_loader.dataset), loss_variances / len(train_loader.dataset), loss_kls / len(train_loader.dataset)

def test(args, loader, model):
    model.eval()
    error = 0
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            predictions, _, _ = model(inputs)
            error += F.mse_loss(predictions, targets, reduction='sum')
        return error / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--lambda-kl', default=1.0, type=float,
                        help="Coefficient for KL loss")
    parser.add_argument('--save-every-n-epochs', default=None, type=int)
    args = parser.parse_args()
    logger.info(dict(args._get_kwargs()))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(dict(args._get_kwargs()), f, indent=4)

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    logging.info(f'Device = {args.device}')

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    train_dataset = PhC2DBandgapQuickLoad(train=True)
    test_dataset = PhC2DBandgapQuickLoad(train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    create_and_train_model(args, train_loader, test_loader)

    args.writer.close()

if __name__ == '__main__':
    main()