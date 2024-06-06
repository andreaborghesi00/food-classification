
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2
import gc


from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, BatchNorm2d, Dropout, Flatten, Sequential, Module, GELU, LeakyReLU, BatchNorm2d
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torchsummary import summary

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import optuna
from optuna.pruners import BasePruner
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# %reload_ext tensorboard
# %tensorboard --logdir={experiment_name}

# %%
if not os.path.exists('dataset'):
    os.makedirs('dataset')

if not os.path.exists('dataset/train_info.csv'):
    os.system('wget https://food-x.s3.amazonaws.com/annot.tar -O dataset/annot.tar')
    os.system('tar -xvf dataset/annot.tar -C dataset')
    os.system('rm dataset/annot.tar')

if not os.path.exists('dataset/train_set'):
    os.system('wget https://food-x.s3.amazonaws.com/train.tar -O dataset/train.tar')
    os.system('tar -xvf dataset/train.tar -C dataset')
    os.system('rm dataset/train.tar')

if not os.path.exists('dataset/test_set'):
    os.system('wget https://food-x.s3.amazonaws.com/test.tar -O dataset/test.tar')
    os.system('tar -xvf dataset/test.tar -C dataset')
    os.system('rm dataset/test.tar')
    
if not os.path.exists('dataset/val_set'):
    os.system('wget https://food-x.s3.amazonaws.com/val.tar -O dataset/val.tar')
    os.system('tar -xvf dataset/val.tar -C dataset')
    os.system('rm dataset/val.tar')


# %%

def get_df(path, class_list=None):
    
    df = pd.read_csv(path, header=None)
    
    if df.shape[1] == 2:
        df.columns = ['image', 'label']
        df['class'] = df['label'].map(class_list['name'])
    else:
        df.columns = ['image']
    return df

class_list = pd.read_csv('dataset/class_list.txt', header=None, sep=' ', names=['class', 'name'], index_col=0)

train_df = get_df('dataset/train_info.csv', class_list)
test_df = get_df('dataset/test_info.csv', class_list)
val_df = get_df('dataset/val_info.csv', class_list)

train_df


# %%
class FoodDataset(Dataset):
        def __init__(self, df, root_dir, transform=None):
            self.df = df
            self.root_dir = root_dir
            self.transform = transform
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
            image = Image.open(img_name)
            
            if self.transform:
                image = self.transform(image)
            
            if self.df.shape[1] == 3:
                label = self.df.iloc[idx, 1]
                return image, label
            else:
                return image


# %%

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
])

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=30),
    transforms.RandomAdjustSharpness(2, p=0.5),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomEqualize(0.5),

])
aug_transform = transforms.Compose([
    augmentation,
    transform
])

train_ds = FoodDataset(train_df, 'dataset/train_set', aug_transform)
test_ds = FoodDataset(test_df, 'dataset/test_set', transform)
val_ds = FoodDataset(val_df, 'dataset/val_set', transform)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)


# %% [markdown]
# ----
# # <center>Neural Networks

# %%
class tinyNet(Module):
    def __init__(self, c1_filters=8, c2_filters=32, c3_filters=64, c4_filters=128, c5_filters=172, fc1_units=256):
        super(tinyNet, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, c1_filters, kernel_size=3, stride=1, padding='same'),
            GELU(),
            Conv2d(c1_filters, c2_filters, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(c2_filters),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = Sequential(
            Conv2d(c2_filters, c2_filters, kernel_size=3, stride=1, padding='same'),
            GELU(),
            Conv2d(c2_filters, c3_filters, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(c3_filters),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = Sequential(
            Conv2d(c3_filters, c3_filters, kernel_size=3, stride=1, padding='same'),
            GELU(),
            Conv2d(c3_filters, c4_filters, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(c4_filters),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv4 = Sequential(
            Conv2d(c4_filters, c4_filters, kernel_size=3, stride=1, padding='same'),
            GELU(),
            Conv2d(c4_filters, c5_filters, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(c5_filters),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv5 = Sequential(
            Conv2d(c5_filters, c5_filters, kernel_size=3, stride=1, padding='same'),
            GELU(),
            Conv2d(c5_filters, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            GELU(),
            MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc1 = Sequential(
            Linear(32*4*4, fc1_units),
            Dropout(.2),
            GELU()
        )

        self.fc2 = Sequential(
            Linear(fc1_units, 251),
            GELU()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 32*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# %%
def train(model, train_dl, val_dl, optimizer, scheduler, criterion, epochs, writer, experiment_name, best_experiment_name, device='cuda'):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    pbar = tqdm(total=epochs)
    n_iter = 0
    best_acc = 0
    best_running_acc = 0
    # ------------------------------ MODEL LOADING ------------------------------
    
    try:
        checkpoint = torch.load(os.path.join('models', 'best_' + best_experiment_name + '.pth'))
        best_model = checkpoint['model']
        best_optimizer = checkpoint['optimizer']
        best_criterion = checkpoint['criterion']
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        print('Best Model loaded, evaluating...')
        best_model.to(device)
        best_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = best_model(inputs)
                loss = best_criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Best model Loss: {running_loss/len(test_dl):.3f}, Test Acc: {100*correct/total:.3f}%')
            best_acc = 100*correct/total
        del best_model, best_criterion, best_optimizer, checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(e)
        print('No best model found, training from scratch...')
        
    
    
    for epoch in range(epochs):
        writer.add_scalar("epoch", epoch, n_iter)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # ------------------------------ TRAINING LOOP ------------------------------
        for i, data in enumerate(train_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            writer.add_scalar("train", loss.item(), n_iter)
            n_iter += 1
            
        train_loss.append(running_loss/len(train_dl))
        train_acc.append(100*correct/total)
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # ------------------------------ VALIDATION LOOP ------------------------------
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                writer.add_scalar("val", loss.item(), n_iter)
        
        # ------------------------------ PRINTING AND MODEL SAVING ------------------------------
        
        pbar.set_description(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.3f}, Train Acc: {train_acc[-1]:.3f}%, Val Loss: {running_loss/len(val_dl):.3f}, Val Acc: {100*correct/total:.3f}%, Acc to beat: {best_acc:.3f}%, best running acc: {best_running_acc:.3f}%')
        val_loss.append(running_loss/len(val_dl))
        val_acc.append(100*correct/total)
        if val_acc[-1] > best_running_acc:
            pbar.set_description(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.3f}, Train Acc: {train_acc[-1]:.3f}%, Val Loss: {running_loss/len(val_dl):.3f}, Val Acc: {100*correct/total:.3f}%, Acc to beat: {best_acc:.3f}%, best running acc beated, saving model')
            best_running_acc = val_acc[-1]
            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'epoch': epoch,
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join('models', 'best_' + experiment_name + '.pth'))
        pbar.update(1)
    pbar.close()
    return max(val_acc)


class MaxParameterPruner(BasePruner):
    def __init__(self, max_params, min_params=0):
        self.max_params = max_params
        self.min_params = min_params

    def prune(self, study, trial):
            # Define the hyperparameters to tune
        c1_filters = trial.suggest_int('num_filters1', 8, 32)
        c2_filters = trial.suggest_int('num_filters2', 16, 64)
        c3_filters = trial.suggest_int('num_filters3', 32, 128)
        c4_filters = trial.suggest_int('num_filters4', 64, 256)
        c5_filters = trial.suggest_int('num_filters5', 64, 256)
        fc1_units = trial.suggest_int('fc1_units', 128, 512)

        # Create the model with the given hyperparameters
        model = tinyNet(c1_filters, c2_filters, c3_filters, c4_filters, c5_filters, fc1_units).to(device)

        # Calculate the number of parameters
        num_params = sum(p.numel() for p in model.parameters())

        # If the number of parameters exceeds 1 million, return a large negative value
        if num_params > self.max_params or num_params < self.min_params:
            study.set_user_attr('num_params', num_params)
            return optuna.exceptions.TrialPruned()


def objective(trial):
    # Define the hyperparameters to tune
    c1_filters = trial.suggest_int('num_filters1', 8, 32)
    c2_filters = trial.suggest_int('num_filters2', 16, 64)
    c3_filters = trial.suggest_int('num_filters3', 32, 128)
    c4_filters = trial.suggest_int('num_filters4', 64, 256)
    c5_filters = trial.suggest_int('num_filters5', 64, 256)
    fc1_units = trial.suggest_int('fc1_units', 128, 512)

    # Create the model with the given hyperparameters

    model = tinyNet(c1_filters, c2_filters, c3_filters, c4_filters, c5_filters, fc1_units).to(device)

    # Calculate the number of parameters

    train_ds = FoodDataset(get_fraction_of_data(train_df, 0.1), 'dataset/train_set', aug_transform)
    val_ds = FoodDataset(get_fraction_of_data(val_df, 0.1), 'dataset/val_set', transform)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0001)
    experiment_name = 'tinyNetHT'
    writer = SummaryWriter('runs/'+experiment_name)


    accuracy = train(model=model,
                     train_dl=train_dl, 
                     val_dl=val_dl, 
                     optimizer=optimizer, 
                     criterion=criterion, 
                     scheduler=scheduler,
                     epochs=epochs, 
                     writer=writer, 
                     experiment_name=experiment_name, 
                     best_experiment_name='tinyNetv2', 
                     device=device)
    
    del model, optimizer, criterion, scheduler, writer, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache()
    gc.collect()

    return accuracy

def get_fraction_of_data(df, fraction, stratified=True):
    if stratified:
        _, train_df = train_test_split(df, test_size=fraction, stratify=df['label'])
    else:
        _, train_df = train_test_split(df, test_size=fraction)
    # return DataLoader(FoodDataset(df.iloc[train_df], 'dataset/train_set', transform), batch_size=128, shuffle=True, num_workers=8)
    return train_df

def run_trial(trial):
    try:
        accuracy = objective(trial)
        return accuracy
    except Exception as e:
        raise optuna.exceptions.TrialPruned()
    
n_trials = 100
study = optuna.create_study(direction='maximize', study_name='tinyNetHT', pruner=MaxParameterPruner(1e6, 9e5))
study.optimize(run_trial, n_trials=n_trials)

study.best_params