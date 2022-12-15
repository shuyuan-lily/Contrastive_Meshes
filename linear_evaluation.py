import os
import argparse
import torch
import numpy as np

from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.classification_data import ClassificationData
from models.logistic_regression import LogisticRegression
from models import create_model
from util.writer import Writer
from data import DataLoader


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []

    for i, data in enumerate(loader):
        simclr_model.set_input(data)
        x = torch.from_numpy(data['edge_features']).to(device)
        y = torch.from_numpy(data['label']).to(device)
        # x, y = data['edge_features'], data['label']
        # print(x.shape, y.shape)
        # get encoding
        with torch.no_grad():
            reps, out = simclr_model.forward()
            h, _ = reps
            z, _ = out

        h = h.detach()
        # print(type(h))
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

        if i % 20 == 0:
            print(f"Step [{i}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(opt, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    accuracy_epoch = 0

    writer.reset_counter()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(opt.device)
        y = y.to(opt.device)
        print(x.shape, y.shape)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer, writer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for i, data in enumerate(loader):
        model.zero_grad()

        x = data['edge_features'].to(opt.device)
        y = data['label'].to(opt.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    print('Running Linear Evaluation')

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''if opt.dataroot == "datasets/shrec_16":
        opt.is_train == True
        train_dataset = ClassificationData(opt)
        opt.is_train == False
        test_dataset = ClassificationData(opt)
    else:
        raise NotImplementedError'''

    opt = TrainOptions().parse()
    assert opt.dataroot == 'datasets/shrec_16', 'Dataset Not Implemented' 
    opt.is_train == True
    simclr_train_loader = DataLoader(opt)
    opt = TestOptions().parse()
    opt.is_train == False
    simclr_test_loader = DataLoader(opt) 
    print("len_train_loader", len(simclr_train_loader))
    print("len_test_loader", len(simclr_test_loader))
    '''train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_threads,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=opt.num_threads,
    )'''
    opt.dataset_mode == 'classification'
    opt.is_train == True
    train_loader = DataLoader(opt)
    opt.is_train == False
    test_loader = DataLoader(opt) 

    opt.dataset_mode == 'simclr'
    # encoder = get_resnet(args.resnet, pretrained=False)
    # n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = create_model(opt)
    writer = Writer(opt)

    ## Logistic Regression
    n_classes = 10  # shrec
    model = LogisticRegression(opt.out_dim, n_classes)
    model = model.to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, simclr_train_loader, simclr_test_loader, opt.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, opt.batch_size
    )

    for epoch in range(opt.lin_eval_num_epoch):
        loss_epoch, accuracy_epoch = train(
            opt, arr_train_loader, simclr_model, model, criterion, optimizer, writer
        )
        print(
            f"Epoch [{epoch}/{opt.lin_eval_num_epoch}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )
        if writer.display:
            writer.display.add_scalar('data/lin_eval_train_loss', loss_epoch / len(arr_train_loader), epoch) 
            writer.display.add_scalar('data/lin_eval_train_acc', accuracy_epoch / len(arr_train_loader), epoch)

    # final testing
    loss_epoch, accuracy_epoch = test(
        opt, arr_test_loader, simclr_model, model, criterion, optimizer, writer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )
    if writer.display:
        writer.display.add_scalar('data/lin_eval_test_loss', loss_epoch / len(arr_test_loader))
        writer.display.add_scalar('data/lin_eval_test_acc', accuracy_epoch / len(arr_test_loader))