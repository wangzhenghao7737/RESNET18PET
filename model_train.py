import copy
import time
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import ResNet18, Residual
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse
from MyModule.SEBlock import SEBlock


def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 with optional SEBlock")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_se', action='store_true', help='Use SEBlock if set')

    return parser.parse_args()


def train_val_data_process(batch_size=32):
    # Define dataset path
    ROOT_TRAIN = './dataset/train'

    normalize = transforms.Normalize([0.48607032,0.45353173,0.4160252 ], [0.06886391,0.06542894,0.0667423 ])
    # Define preprocessing steps for dataset
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # Load dataset
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.9*len(train_data)), round(0.1*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader



def train_model_process(model, train_dataloader, val_dataloader, num_epochs=50, lr=0.001, save_name="undefined"):
    # Set training device, use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use Adam optimizer with learning rate 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Use cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    # Move model to training device
    model = model.to(device)
    # Copy initial model weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize parameters
    # Best accuracy
    best_acc = 0.0
    # Training loss list
    train_loss_all = []
    # Validation loss list
    val_loss_all = []
    # Training accuracy list
    train_acc_all = []
    # Validation accuracy list
    val_acc_all = []

    for epoch in range(num_epochs):
        since = time.time()

        # Initialize parameters
        # Training loss
        train_loss = 0.0
        # Training accuracy
        train_corrects = 0
        # Validation loss
        val_loss = 0.0
        # Validation accuracy
        val_corrects = 0
        # Number of training samples
        train_num = 0
        # Number of validation samples
        val_num = 0

        # Train and evaluate on each mini-batch
        for step, (b_x, b_y) in tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc=f"Epoch Train {epoch + 1}/{num_epochs}"):
            # Move inputs to training device
            b_x = b_x.to(device)
            # Move labels to training device
            b_y = b_y.to(device)
            # Set model to training mode
            model.train()

            # Forward pass: predict outputs for current batch
            output = model(b_x)
            # Get predicted class indices
            pre_lab = torch.argmax(output, dim=1)
            # Compute loss for the batch
            loss = criterion(output, b_y)

            # Zero gradients
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            # Update parameters to reduce loss
            optimizer.step()
            # Accumulate loss
            train_loss += loss.item() * b_x.size(0)
            # Count correct predictions
            train_corrects += torch.sum(pre_lab == b_y.data)
            # Count total training samples
            train_num += b_x.size(0)
        for step, (b_x, b_y) in tqdm(enumerate(val_dataloader),total=len(val_dataloader), desc=f"Epoch val {epoch + 1}/{num_epochs}"):
            # Move inputs to validation device
            b_x = b_x.to(device)
            # Move labels to validation device
            b_y = b_y.to(device)
            # Set model to evaluation mode
            model.eval()
            # Forward pass
            output = model(b_x)
            # Get predicted class indices
            pre_lab = torch.argmax(output, dim=1)
            # Compute loss
            loss = criterion(output, b_y)

            # Accumulate loss
            val_loss += loss.item() * b_x.size(0)
            # Count correct predictions
            val_corrects += torch.sum(pre_lab == b_y.data)
            # Count total validation samples
            val_num += b_x.size(0)

        # Compute and save loss and accuracy for each epoch
        # Save training loss
        train_loss_all.append(train_loss / train_num)
        # Save training accuracy
        train_acc_all.append(train_corrects.double().item() / train_num)

        # Save validation loss
        val_loss_all.append(val_loss / val_num)
        # Save validation accuracy
        val_acc_all.append(val_corrects.double().item() / val_num)

        time_use = time.time() - since
        print('\n' + """Epoch {} Time {:.4f} m {:.4f}s
                Train Loss: {:.4f} Train Acc: {:.4f}
                Val Loss: {:.4f} Val Acc: {:.4f}
                """.format(epoch + 1, time_use // 60, time_use % 60,
                           train_loss_all[-1], train_acc_all[-1],
                           val_loss_all[-1], val_acc_all[-1]))
        if val_acc_all[-1] > best_acc:
            # Save best accuracy
            best_acc = val_acc_all[-1]
            # Save weights of the best-performing model
            best_model_wts = copy.deepcopy(model.state_dict())

    folder_path = "./model"
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]) + 1
    torch.save(best_model_wts, f"./model/{save_name}-{file_count}.pth")
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    return train_process, f"./model/best_model{file_count}.pth"


def matplot_acc_loss(train_process,png_name="undefined"):
    # Display training and validation loss/accuracy for each epoch
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    # plt.show()
    plt.savefig(f"{png_name}.png")


if __name__ == '__main__':
    args = get_args()
    now_parameters = f"epochs:{args.epochs}-batch_size:{args.batch_size}-lr:{args.lr}-SEBlock:{args.use_se}".replace('.', '_')
    attention = SEBlock if args.use_se else None
    model = ResNet18(Residual, attention=attention)
    train_data, val_data = train_val_data_process(batch_size=args.batch_size)
    train_process, src = train_model_process(model, train_data, val_data, num_epochs=args.epochs, lr=args.lr, save_name=now_parameters)
    # python model_train.py --epochs 50 --batch_size 64 --lr 0.0005 --use_se
    png_name = "process_data"+now_parameters
    matplot_acc_loss(train_process,png_name)
    print(f"Finish:{src}")
