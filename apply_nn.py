#Programm applying pre-generated NN on a set of cropped face fotos
#By Jarov Alexey Valeryevich
#Returns a JSON file with predictions of a gender

#Importing the necessary packages
import imutils
from imutils import paths
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from keras.preprocessing.image import img_to_array
import json

#Preparing preprocess of images
def preprocess(image, width, height):
    image = cv2.resize(image, (width, height))
    return image

#Preparing export to JSON
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)

#Inputing path to dataset for analys and pre-generated NN
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

#Defining labels, data array and counter
classLabels = ["female", "male"]
data = []
i = 0

#Loading images to dataset
print("[INFO] loading dataset...")
for imagePath in paths.list_images(args["dataset"]):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

#Scaling the raw pixel intensities to the range [0, 1]
print("[INFO] scaling dataset...")
data = np.array(data, dtype="float") / 255.0

#Loading Network
print("[INFO] loadig pre-trained network...")
model = load_model(args["model"])

#Making predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

#Looping over the sample images
data = {}
for imagePath in paths.list_images(args["dataset"]):
#Drawing the prediction, and adding it to array
    data[imagePath] = classLabels[preds[i]]
    i += 1

#Exporting to JSON file
writeToJSONFile('./','process_results',data)
