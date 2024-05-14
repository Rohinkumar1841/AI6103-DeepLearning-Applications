import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from mobilenet import MobileNet
from utils import plot_loss_acc
from collections import Counter
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import random
import argparse
from pdf_plot import plot_beta_distribution_pdf

def get_train_valid_loader(dataset_dir, batch_size, shuffle, seed, save_images=False):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(([0.4385, 0.4181, 0.3775]), ([0.3004, 0.2872, 0.2937]))
    ])

    # Load the CIFAR-100 dataset
    full_dataset = datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)

    torch.manual_seed(seed)

    # Randomly split the dataset into training and validation sets
    train_size = 40000
    valid_size = 10000
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader

def get_test_loader(dataset_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(([0.4385, 0.4181, 0.3775]), ([0.3004, 0.2872, 0.2937]))
    ])

  
    test_dataset = datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform)


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

# Set the random seed
random.seed(0)

def mixup_data(data, targets, alpha):
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    batch_size = data.size(0)
    index = torch.randperm(batch_size)
    mixed_data = lam * data + (1 - lam) * data[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]
    return mixed_data, mixed_targets


#
def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders youself
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=args.save_images) 
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    # model
    model = MobileNet(100)
    print(model)
    model.cuda()

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    for epoch in range(args.epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        for imgs, labels in train_loader:
            if args.mixup:
                imgs, labels = mixup_data(imgs, labels, alpha = 0.02)

        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            logits = model.forward(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, top_class = logits.topk(1, dim=1)
            equals=top_class==labels.view(*top_class.shape)
            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size
        
        # validation
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        # print
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        # lr scheduler
        scheduler.step()
    # plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
    plot_beta_distribution_pdf(alpha=0.2, fig_name='beta_pdf_alpha_0.2.png')
    # test
    if args.test:
        test_loss_total = 0
        test_acc = 0
        test_samples = 0
        
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            current_batch_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss_total += batch_size * current_batch_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        print('Test loss: ', test_loss_total / test_samples)
        print('Test acc: ', test_acc / test_samples)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False )
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(mixup=False)
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--mixup', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    main(args)
