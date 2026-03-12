import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from paz.models import MiniXception
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

csvFile = r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'
batchSize = 32
epochs = 50
learningRate = 0.001
imageSize = 48
numClasses = 7

if not os.path.exists(csvFile):
    print("file missing")
    exit(1)

print("file located")
print("loading dataset")

data = pd.read_csv(csvFile)
print("samples loaded:", len(data))
print("columns available:", list(data.columns))

def convertToImages(dataframe):
    images = []
    labels = []
    
    for index, row in dataframe.iterrows():
        pixelValues = row['pixels'].split()
        pixelArray = np.array(pixelValues, dtype=np.uint8)
        
        image = pixelArray.reshape(imageSize, imageSize, 1)
        image = image.astype('float32') / 255.0
        
        images.append(image)
        labels.append(int(row['emotion']))
        
        if (index + 1) % 5000 == 0:
            print("processed samples")
    
    xData = np.array(images)
    yData = to_categorical(np.array(labels), numClasses)
    
    return xData, yData

trainData = data[data['Usage'] == 'Training']
testData = data[data['Usage'] != 'Training']

print("split information")
print("training samples:", len(trainData))
print("testing samples:", len(testData))

print("converting training")
xTrain, yTrain = convertToImages(trainData)

print("converting test")
xTest, yTest = convertToImages(testData)

print("preparation complete")
print("training shape:", xTrain.shape)
print("labels shape:", yTrain.shape)
print("test shape:", xTest.shape)
print("test labels:", yTest.shape)

print("configuring augmentation")
augmentationGenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

print("initializing model")
model = MiniXception(input_shape=(imageSize, imageSize, 1), 
                     num_classes=numClasses)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learningRate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("model summary")
model.summary()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experimentFolder = 'experiments/FER_Baseline_' + timestamp
os.makedirs(experimentFolder, exist_ok=True)

print("saving to:", experimentFolder)

checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(experimentFolder, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

csvLogger = keras.callbacks.CSVLogger(
    filename=os.path.join(experimentFolder, 'training_history.csv'),
    separator=',',
    append=False
)

earlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduceLr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacksList = [checkpointCallback, csvLogger, earlyStopping, reduceLr]

print("starting training")
print("epochs total:", epochs)
print("batch size:", batchSize)

trainingHistory = model.fit(
    augmentationGenerator.flow(xTrain, yTrain, batch_size=batchSize),
    steps_per_epoch=len(xTrain) // batchSize,
    epochs=epochs,
    validation_data=(xTest, yTest),
    callbacks=callbacksList,
    verbose=1
)

print("evaluating model")
testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose=0) #####

print("training completed")
print("test accuracy:", round(testAccuracy * 100, 2), "%")
print("test loss:", round(testLoss, 4))

finalModelPath = os.path.join(experimentFolder, 'final_model.h5')
model.save(finalModelPath)
print("model saved:", finalModelPath)

print("generating plots")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(trainingHistory.history['accuracy'], label='training accuracy')
axes[0].plot(trainingHistory.history['val_accuracy'], label='validation accuracy')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('accuracy')
axes[0].set_title('model accuracy over training')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(trainingHistory.history['loss'], label='training loss')
axes[1].plot(trainingHistory.history['val_loss'], label='validation loss')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss')
axes[1].set_title('model loss over training')
axes[1].legend()
axes[1].grid(True)

plotPath = os.path.join(experimentFolder, 'training_curves.png')
plt.tight_layout()
plt.savefig(plotPath, dpi=300)
print("plots saved:", plotPath)

summaryPath = os.path.join(experimentFolder, 'experiment_summary.txt')
with open(summaryPath, 'w') as f:
    f.write("FER2013 emotion recognition baseline experiment\n")
    f.write("=" * 60 + "\n\n")
    f.write("date: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("dataset=>FER2013\n")
    f.write("model: MiniXception\n\n")
    f.write("training configuration:\n")
    f.write("epochs=>" + str(epochs) + "\n")
    f.write("batch size=>" + str(batchSize) + "\n")
    f.write("learning rate=>" + str(learningRate) + "\n")
    f.write("image size=>" + str(imageSize) + "x" + str(imageSize) + "\n")
    f.write("number of classes=>" + str(numClasses) + "\n\n")
    f.write("data augmentation=>\n")
    f.write("rotation range=> 30 degrees\n")
    f.write("width shift=> 10%\n")
    f.write("height shift=>10%\n")
    f.write("zoom range=>10%\n")
    f.write("horizontal flip= yes\n\n")
    f.write("Results\n")
    f.write("training samples=>" + str(len(trainData)) + "\n")
    f.write("test samples=>" + str(len(testData)) + "\n")
    f.write("final test accuracy: " + str(round(testAccuracy * 100, 2)) + "%\n")
    f.write("final test loss: " + str(round(testLoss, 4)) + "\n")
    f.write("best validation accuracy: " + str(round(max(trainingHistory.history['val_accuracy']) * 100, 2)) + "%\n")

print("summary saved:", summaryPath)
print("artifacts saved")
print("baseline complete")