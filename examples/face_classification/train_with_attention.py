# Training with Squeeze-and-Excitation attention blocks
import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # FIXED
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

csvfile = r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'
batchsize = 32
epochs = 30  
learningrate = 0.001
imagesize = 48
numclasses = 7



# squeeze and excitation block
def seblock(inputtensor, ratio=16):
   
    channels = inputtensor.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(inputtensor)
    
    excitation = layers.Dense(channels // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    excitation = layers.Reshape((1, 1, channels))(excitation)
    scaled = layers.Multiply()([inputtensor, excitation])
    
    return scaled

# custom cnn architecture with attention
def buildCNNWithAttention(inputshape, numclasses):
    inputs = layers.Input(shape=inputshape)
    
    x = layers.Conv2D(32,(3, 3),activation='relu',padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = seblock(x)  
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    

    x = layers.Conv2D(64,(3, 3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = seblock(x)  
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128,(3, 3),activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = seblock(x) 
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(numclasses, activation='softmax')(x)
    
    return keras.Model(inputs=inputs,outputs=outputs,name='CNN_with_Attention')

print("loading dataset")
data = pd.read_csv(csvfile)


# data preprocessing
def converttoimages(dataframe):
    images = []
    labels = []
    for index, row in dataframe.iterrows():
        pixelvalues = row['pixels'].split()
        pixelarray = np.array(pixelvalues, dtype=np.uint8)
        image = pixelarray.reshape(imagesize, imagesize, 1)
        image = image.astype('float32') / 255.0
        images.append(image)
        labels.append(int(row['emotion']))
        if (index + 1) % 5000 == 0:
            print(f"procesd {index + 1} samples")
    return np.array(images), to_categorical(np.array(labels), numclasses)

trainData = data[data['Usage'] == 'Training']
testData = data[data['Usage'] != 'Training']

print("converting data")
xTrain, yTrain = converttoimages(trainData)
xTest, yTest = converttoimages(testData)


augmentationgenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

print("building model with attention mechanism")
model = buildCNNWithAttention((imagesize, imagesize, 1), numclasses)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learningrate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("model summary")
model.summary()

# Training
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experimentfolder = 'experiments/FER_Attention_' + timestamp
os.makedirs(experimentfolder, exist_ok=True)


# training callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(experimentfolder, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        os.path.join(experimentfolder, 'training_history.csv')
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]



# model training
print("starting training with attention mechanism")
trainingHistory = model.fit(
    augmentationgenerator.flow(xTrain, yTrain, batch_size=batchsize),
    steps_per_epoch=len(xTrain) // batchsize,
    epochs=epochs,
    validation_data=(xTest, yTest),
    callbacks=callbacks,
    verbose=1
)


# model evaluation
testloss, testaccuracy = model.evaluate(xTest, yTest, verbose=0)
print(f"test accuracy: {testaccuracy*100:.2f}%")
print(f"test loss: {testloss:.4f}")


model.save(os.path.join(experimentfolder, 'final_model.h5'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(trainingHistory.history['accuracy'], label='training accuracy')
axes[0].plot(trainingHistory.history['val_accuracy'], label='validation accuracy')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('accuracy')
axes[0].set_title('model accuracy')
axes[0].legend()
axes[0].grid(True)


# loss and save plots
axes[1].plot(trainingHistory.history['loss'], label='training loss')
axes[1].plot(trainingHistory.history['val_loss'], label='validation loss')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss')
axes[1].set_title('model loss')
axes[1].legend()
axes[1].grid(True)

plotpath = os.path.join(experimentfolder, 'training_curves.png')
plt.tight_layout()
plt.savefig(plotpath, dpi=300)
print("plots", plotpath)

with open(os.path.join(experimentfolder, 'summary.txt'), 'w') as f:
    f.write("model with attention mechanism\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Accuracy: {testaccuracy*100:.2f}%\n")
    f.write(f"Test Loss: {testloss:.4f}\n")
    f.write(f"Epochs trained: {len(trainingHistory.history['accuracy'])}\n")
    f.write(f"Best validation accuracy: {max(trainingHistory.history['val_accuracy'])*100:.2f}%\n\n")
print("training complete")










