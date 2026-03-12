import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.models import MiniXception
from tensorflow import keras
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

csvfile = r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'
batchsize = 32
epochs = 30  
learningrate = 0.001
imagesize = 48
numclasses = 7

print("loading dataset")
data = pd.read_csv(csvfile)


# histogram equalization preprocessing
def converttoimageswithhistogrameq(dataframe):
   
    images = []
    labels = []
    
    for index, row in dataframe.iterrows():
        pixelvalues = row['pixels'].split()
        pixelarray = np.array(pixelvalues, dtype=np.uint8)
        image = pixelarray.reshape(imagesize, imagesize)
        

        imageequalized = cv2.equalizeHist(image)
        
        imagenormalized = imageequalized.astype('float32') / 255.0
        imagenormalized = imagenormalized.reshape(imagesize, imagesize, 1)
        
        images.append(imagenormalized)
        labels.append(int(row['emotion']))
        
        if (index + 1) % 5000 == 0:
            print(f"processed {index + 1} samples with histogram equalisation")
    
    return np.array(images), to_categorical(np.array(labels), numclasses)

# train-test split
traindata = data[data['Usage'] == 'Training']
testdata = data[data['Usage'] != 'Training']

print("converting with histogram equalization")
xTrain, yTrain = converttoimageswithhistogrameq(traindata)
xTest, yTest = converttoimageswithhistogrameq(testdata)

# Same augmentation as baseline, only preprocessing differs
augmentationGenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

model = MiniXception((imagesize, imagesize, 1), numclasses)


# Compile with same settings as baseline

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learningrate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experimentfolder = 'experiments/FER_Histogram_' + timestamp
os.makedirs(experimentfolder, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(experimentfolder, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.CSVLogger(
        os.path.join(experimentfolder, 'training_history.csv')
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

print("training with histogram equalization")
traininghistory = model.fit(
    augmentationGenerator.flow(xTrain, yTrain, batch_size=batchsize),
    steps_per_epoch=len(xTrain) // batchsize,
    epochs=epochs,
    validation_data=(xTest, yTest),
    callbacks=callbacks,
    verbose=1
)

testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose=0)
print(f"test accuracy: {testAccuracy*100:.2f}%")

model.save(os.path.join(experimentfolder, 'final_model.h5'))

with open(os.path.join(experimentfolder, 'summary.txt'), 'w') as f:
    f.write("Histogram Equalization\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"test Accuracy: {testAccuracy*100:.2f}%\n")
    f.write(f"test Loss: {testLoss:.4f}\n")
    f.write("\nmodification: applied histogram equalization (Lecture 3)\n")

print("complete")

















