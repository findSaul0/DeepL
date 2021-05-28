from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet34
import numpy as np


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229,0.244, 0.225]
    )
])

train_set = ImageFolder('Cleopatra_Dataset/Fragmented_artworks/Train_set', transform= transform)
valid_set = ImageFolder('Cleopatra_Dataset/Fragmented_artworks/Validation_set', transform= transform)
train_loder = DataLoader(train_set,batch_size=64, shuffle=True)
valid_loder = DataLoader(valid_set,batch_size=64, shuffle=True)

cnn = resnet34(pretrained=True)
cnn.eval()
print(cnn)


images,_ = next(iter(train_loder))
save_image(make_grid(images,nrow=8),'batch.png')
im = Image.open('batch.png')
im.show()


### FOR TEST
X_train = []
y_train = []
for images, labels in train_loder:
    X_train.append(images.view(images.size(0), -1).numpy())
    y_train.append(labels.numpy().flatten())

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
print(X_train.shape)
print(y_train.shape)





### FOR VALIDATION
X_valid = []
y_valid = []
for images, labels in valid_loder:
    X_valid.append(images.view(images.size(0), -1).numpy())
    y_valid.append(labels.numpy().flatten())

X_valid = np.vstack(X_valid)
y_valid = np.hstack(y_valid)
print(X_valid.shape)
print(y_valid.shape)



#pca = PCA(n_components=0.98)
#X_train_t = pca.fit_transform(X_train)
#X_valid_t = pca.fit_transform(X_valid)
#print (X_t.shape)

#svm = LinearSVC()
#svm.fit(X,y)
#svm.score(X,y)

#rf = RandomForestClassifier(n_estimators = 20, max_depth=5)
#rf.fit(X_train_t , y_train)
#print(rf.score(X_train_t, y_train))
#print(rf.score(X_valid_t, y_valid))
