import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# constants
#img_dir = "./data/"
img_dir = "./newtrainingdata/"

# This is key to getting it right
steering_offset = 0.21


# Let us get the images and steering angles into 2 dataframes - X_all and y_all 
drive_log = pd.read_csv(img_dir+'driving_log.csv', names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'], header=None)

center = []
left = []
right = []
steering = drive_log['Steering Angle'].astype(np.float32)

for i in range(len(drive_log)):
    center.append(drive_log['Center'][i].split('IMG/')[1])
    left.append(drive_log['Left'][i].split('IMG/')[1])
    right.append(drive_log['Right'][i].split('IMG/')[1])


df = pd.DataFrame(steering)

# X_all is the path of images, y_all is the steering angle
X_all = np.concatenate((center, left, right), axis = 0)
y_all = np.concatenate((df, df+steering_offset, df-steering_offset), axis = 0)

X_all, y_all = shuffle(X_all, y_all)
print("Total samples of images and steering angles: {}, {}".format(len(X_all), len(y_all)))


# Let us split in 80:20 training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2)

total_train_samples = len(X_train)
total_val_samples = len(X_val)
print("Total samples for training and validation: {}, {}".format(total_train_samples, total_val_samples))



# Utility functions
def crop_image(image):
    return image[60:140, 20:300]

# Flipping helps add more samples (mirror image), hence train the network better
def flip_image(image, steering_angle):
    return cv2.flip(image, 1), -1.0 * steering_angle


# Generate training and validation batch
# The idea is to feed the images in 1 batch at a time into the training model, instead of loading the entire set of images into memory
# Note that we're generating batch_size*2 worth of samples every batch - 128 in this case
# we are not using cropped images because we'll do that in Keras

tp = 0

def generate_training_batch(batch_size=64):
    global tp
    
    while 1:
        X_batch = []
        y_batch = []
        
        # pull 'batch_size' worth of data starting at 'tp'. xb is images. yb is steering angles.
        xb = X_train[tp:tp+batch_size]
        yb = y_train[tp:tp+batch_size]
        
        # increment tp. wrap-around if > len(X_train)
        if tp + batch_size >= len(X_train):
            tp = 0
        else:
            tp += batch_size
        
        # At this point we need to pass xb, yb through the processing pipeline
        for i in range(len(xb)):
            img = cv2.imread(img_dir + 'IMG/' + xb[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ang = yb[i]
            X_batch.append(img)
            y_batch.append(ang)
            # Also added the flipped image to the batch
            img, ang = flip_image(img, ang)
            X_batch.append(img)
            y_batch.append(ang)
                
                
        yield shuffle(np.array(X_batch), np.array(y_batch))


# Generate validation batch
vp = 0

def generate_validation_batch(batch_size=64):
    global vp
    
    while 1:
        X_batch = []
        y_batch = []
        
        # pull 'batch_size' worth of data starting at 'tp'. xb is images. yb is steering angles.
        xb = X_val[vp:vp+batch_size]
        yb = y_val[vp:vp+batch_size]
        
        # increment vp. wrap-around if > len(X_train)
        if vp + batch_size >= len(X_val):
            vp = 0
        else:
            vp += batch_size
        
        # At this point we need to pass xb, yb through the processing pipeline
        for i in range(len(xb)):
            img = cv2.imread(img_dir + 'IMG/' + xb[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ang = yb[i]
            X_batch.append(img)
            y_batch.append(ang)
            # Also added the flipped image to the batch
            img, ang = flip_image(img, ang)
            X_batch.append(img)
            y_batch.append(ang)
                
        yield np.array(X_batch), np.array(y_batch)


# Training model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam


# Hyper-parameters. Going beyond 5 epochs caused overfilling (validation error increased) for my training data
learning_rate = 0.00005
keep_prob = 0.5
no_of_epochs = 5


# NVIDIA model
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (20,20)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


print("Training model summary: {}".format(model.summary()))
model.compile(optimizer=Adam(learning_rate), loss="mse" )

train_data_gen = generate_training_batch()
validation_data_gen = generate_validation_batch()

history = model.fit_generator(train_data_gen,
                              samples_per_epoch=total_train_samples*2,
                              nb_epoch=no_of_epochs,
                              validation_data=validation_data_gen,
                              nb_val_samples=total_val_samples*2,
                              verbose=1)


model.save('./model.h5')
print("Model is saved to model.h5 file in the current directory")

# Test model
num_images = 100
n_train = len(X_train)
indices = np.random.choice(list(range(n_train)), size=num_images, replace=False)

# Obtain the images and labels
images = X_train[indices]
labels = y_train[indices]

print("Now testing the model in 100 random images")

for i, image in enumerate(images):
    img = cv2.imread(img_dir + 'IMG/' + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    angl = labels[i]
    predicted_angl = model.predict(img[None,:,:,:], batch_size=1)
    print("Trained steering angle: {}, Predicted steering angle: {}".format(angl, predicted_angl))


print("Model is ready in model.h5 file!")
