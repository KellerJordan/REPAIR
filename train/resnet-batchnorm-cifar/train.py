import os
import uuid
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision.transforms as T

from model import resnet20
from data import get_loaders

# evaluates accuracy
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
    return correct

def main():

    train_aug_loader, _, test_loader = get_loaders()

    model = resnet20(args.width).cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    ne_iters = len(train_aug_loader)
    warmup = np.interp(np.arange(1+5*ne_iters), [0, 5*ne_iters], [1e-6, 1])
    ni = (args.epochs-5)*ne_iters
    xx = np.arange(ni)/ni
    cosine = (np.cos(np.pi*xx) + 1)/2
    lr_schedule = np.concatenate([warmup, cosine])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()
    
    losses = []
    for _ in tqdm(range(args.epochs)):
        model.train()
        for i, (inputs, labels) in enumerate(train_aug_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs.cuda())
                loss = loss_fn(outputs, labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        print('Acc=%.2f%%' % (evaluate(model, loader=test_loader)/100))

    sd = model.state_dict()
    torch.save(sd, './checkpoints/batchnorm_resnet20x%d_e%d_%s.pt' % (args.width, args.epochs, str(uuid.uuid4())))

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.4)
if __name__ == '__main__':
    args = parser.parse_args()
    main()

