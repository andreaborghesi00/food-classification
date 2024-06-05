# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ----
# # <center>Dataset Preprocessing
#

# %%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2


from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

train_ds = FoodDataset(train_df, 'dataset/train_set', transform)
test_ds = FoodDataset(test_df, 'dataset/test_set', transform)
val_ds = FoodDataset(val_df, 'dataset/val_set', transform)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8)


# %% [markdown]
# ----
# # <center>Neural Networks

# %%
class simple_CNN(torch.nn.Module):
    def __init__(self):
        super(simple_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64*8*8, 256)  # 8*8 is the size of the image after 4 maxpooling layers
        self.fc2 = torch.nn.Linear(256, 251)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.pool(self.gelu(self.conv1(x)))
        x = self.pool(self.gelu(self.conv2(x)))
        x = self.pool(self.gelu(self.conv3(x)))
        x = self.pool(x)  # Additional pooling layer
        x = x.view(-1, 64*8*8)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
def train(model, train_dl, val_dl, optimizer, criterion, epochs):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss.append(running_loss/len(train_dl))
        train_acc.append(100*correct/total)
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
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
                
        val_loss.append(running_loss/len(val_dl))
        val_acc.append(100*correct/total)
        
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.3f}, Train Acc: {train_acc[-1]:.3f}%, Val Loss: {val_loss[-1]:.3f}, Val Acc: {val_acc[-1]:.3f}%')


# %%
model = simple_CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
print(f'the model has {sum(p.numel() for p in model.parameters())} parameters')

# %%
train(model, train_dl, val_dl, optimizer, criterion, epochs)

# %% [markdown]
# ----
# # <center>SIFT and Bag of Words for feature extraction

# %%
force_recompute = False

def extract_sift_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return np.empty((0, 128), dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize image
    gray = cv2.resize(gray, (128, 128))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        # No descriptors found, return an empty array
        descriptors = np.empty((0, 128), dtype=np.float32)
    return descriptors

def extract_sift_features_from_df(df, root_dir):
    features = []
    for i in tqdm(range(len(df))):
        img_name = os.path.join(root_dir, df.iloc[i, 0])
        features.append(extract_sift_features(img_name))
    return features



# %%
def extract_bag_of_words(features, dictionary):
    bow = []
    for i in tqdm(range(len(features))):
        if features[i].size > 0:  # Skip if features[i] is empty
            words = dictionary.predict(features[i].astype(np.float32))
            bow.append(np.bincount(words, minlength=dictionary.n_clusters))
        else:
            bow.append(np.zeros(dictionary.n_clusters, dtype=np.float32))  # Append zeros if no features
    return np.array(bow)


# %%
force_recompute_sift = False
force_recompute_bow = False

if not os.path.exists('dataset/train_features.pkl') or force_recompute_sift:
    train_features = extract_sift_features_from_df(train_df, 'dataset/train_set')
    pickle.dump(train_features, open('dataset/train_features.pkl', 'wb'))
    test_features = extract_sift_features_from_df(test_df, 'dataset/test_set')
    pickle.dump(test_features, open('dataset/test_features.pkl', 'wb'))
    val_features = extract_sift_features_from_df(val_df, 'dataset/val_set')
    pickle.dump(val_features, open('dataset/val_features.pkl', 'wb'))
else:
    train_features = pickle.load(open('dataset/train_features.pkl', 'rb'))
    val_features = pickle.load(open('dataset/val_features.pkl', 'rb'))


n_clusters = 1000

if not os.path.exists('dataset/dictionary.pkl') or force_recompute_bow:
    dictionary = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    dictionary.fit(np.concatenate(train_features))
    pickle.dump(dictionary, open('dataset/dictionary.pkl', 'wb'))
else:
    dictionary = pickle.load(open('dataset/dictionary.pkl', 'rb'))


# %%
train_bow = extract_bag_of_words(train_features, dictionary)
del train_features
val_bow = extract_bag_of_words(val_features, dictionary)
del val_features

# %%
# SVM
clf = SVC()
clf.fit(train_bow, train_df.iloc[:, 1], )
train_pred = clf.predict(train_bow)
val_pred = clf.predict(val_bow)
# save to file
pickle.dump(train_pred, open('dataset/svm_train_pred.pkl', 'wb'))
pickle.dump(val_pred, open('dataset/svm_val_pred.pkl', 'wb'))




train_acc = accuracy_score(train_df.iloc[:, 1], train_pred)
val_acc = accuracy_score(val_df.iloc[:, 1], val_pred)

print(f'SVM Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f}')

# %% [markdown]
# ----
# # <center>Playground

# %%
# extract sift from 1/4 of the images in the training set
#train_14_features = extract_sift_features_from_df(train_df.iloc[::4], 'dataset/train_set')

# %%
# extract bow from 1/4 of the images in the training set
#dictionary = MiniBatchKMeans(n_clusters=1000, random_state=0)
#ictionary.fit(np.concatenate(train_14_features))
#train_14_bow = extract_bag_of_words(train_14_features, dictionary)

# %%
# visualize the bow of the first image
plt.bar(range(len(train_14_bow[0])), train_14_bow[0])
plt.xlabel('Visual Word Index')
plt.ylabel('Frequency')
plt.title('Bag of Words')
plt.show()

