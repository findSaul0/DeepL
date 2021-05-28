import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import normalize, to_categorical
from keras.layers import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

# DIMENSIONE IMG
img_widht , img_height = 150,150


#
#(X_train , y_train), (X_test, y_test) = "Frag/Train_set" , "Frag/Test_set_Cleopatra"

#X_train = normalize(X_train, axis = 1)
#X_test = normalize(X_test,axis=1)

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)


train_data_dir = "Frag/Train_set"
validation_data_dir = "Frag/Validation_set"

nb_train_samples = 800
nb_validation_samples = 100
epochs = 50
batch_size = 20

if K.image_data_format() =='channels_first':
    input_shape = (3,img_widht,img_height)
else:
    input_shape = (img_widht,img_height,3)


train_datagen = ImageDataGenerator(
    rescale=1. /255,
    #rotation_range=45,
    #width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = "Frag/Test_set_Cleopatra"

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_widht,img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_widht,img_height),
    batch_size=batch_size,
    class_mode='binary'
)

#Neural Network
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs= epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

img_pred = image.load_img('Frag/Validation_set/Cubism/10.27.1.png', target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

rslt = model.predict(img_pred)
print(rslt)
# 1 = cubismo
# 0 = biz
if rslt[0][0] == 1:
    prediction = "cubism"
else:
    prediction = "byzantine"

print(prediction)

