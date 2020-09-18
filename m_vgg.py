#Programm creating a neuro-net capable of distinguising between genders by a cropped face photo
#By Jarov Alexey Valeryevich
#Implementing "Mini-VGG" architecture by doctor Adrian Rosebrook
#Code based on some examples in his work "Deep learning for Computer Vision with Python"

#Importing the necessary packages
import cv2
import imutils
from imutils import paths
import argparse
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
import numpy as np

#Defining net's architecture
class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initializing the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # Returning the constructed network architecture
        return model

#Simplifiyng the resize process
def preprocess(image, width, height):
    image = cv2.resize(image, (width, height))
    return image

#Supporting starting parameters: path to learning dataset and title of resulting NN file
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

#There are two arrays for image data and associated labels
data = []
labels = []
#Because of my comparably weak processor and lack of CUDA support resulted in dataset of 10000 images,
#I added an additional variable to monitor process of transforming images to array, that line had stated:
#i = 0
#You are free to de-comment the line upper if you wish. I was just wery eager!

#Looping over dataset images, preprocessing them and adding to data array:
print("[INFO] loading dataset...")

for imagePath in paths.list_images(args["dataset"]):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

#If you wish to get a notice about number of inputed images to moment,
#De-comment two lines below:
#    i += 1
#    if i % 500 == 0:
#        print(i)

#Scaling the raw pixel intensities to the range [0, 1]
print("[INFO] scaling dataset...")
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#Converting the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

#Partitioning the data into training and testing splits using 75% of
#the data for training and the remaining 25% for testing
print("[INFO] splitting dataset...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#Initializing the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#Training the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=15, verbose=1)

#Evaluating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

#Saving the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
