import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

csvfile = r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'
data = pd.read_csv(csvfile)
testdata = data[data['Usage'] != 'Training']

imagesize = 48
numclasses = 7
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Initialize lists to store processed images and labels
xTest = []
yTest = []

# Process each test sample
for index, row in testdata.iterrows():
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    image = pixels.reshape(imagesize, imagesize, 1) / 255.0
    xTest.append(image)
    yTest.append(int(row['emotion']))

# Convert lists to numpy arrays

xTest = np.array(xTest)
ytestlabels = np.array(yTest)
print("loading baseline model")
model = load_model('experiments/FER_Baseline_20260202_191825/best_model.h5')


# generate predictions
predictions = model.predict(xTest, verbose=1)
predictedlabels = np.argmax(predictions, axis=1)
cm = confusion_matrix(ytestlabels, predictedlabels)

# confusion matrix visualization
plt.figure(figsize=(10, 8))


# create heatmap using seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS,
            yticklabels=EMOTIONS)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)

# save report to file for documentation
with open('classreport.txt', 'w') as f:
    f.write(classification_report(ytestlabels, predictedlabels, 
                                 target_names=EMOTIONS))
































# code 2
"""
================================================================================
COSC-4427: Computer Vision Project
Confusion Matrix and Error Analysis - Model Evaluation

Author: Student 3 (Evaluation & Demo Lead)
Date: February 2026
Course: Computer Vision (Winter 2026)
Professor: Omar Al-Buraiki

Description:
This script generates a comprehensive error analysis of our baseline emotion
recognition model. It creates a confusion matrix to visualize which emotions
are frequently confused with each other, and produces a detailed classification
report with precision, recall, and F1-score for each emotion class.

Purpose:
- Identify model strengths and weaknesses
- Understand which emotions are hard to distinguish
- Guide future improvements (e.g., focus on confused pairs)
- Demonstrate scientific analysis and critical thinking

Key Insights Expected:
- Fear vs Surprise confusion (both have wide eyes)
- Sad vs Neutral confusion (subtle differences)
- Happy: High accuracy (clear smile feature)
- Disgust: Low accuracy (minority class, only 1.5% of data)

Output Files:
- confusion_matrix_baseline.png: Visual heatmap of confusion
- classification_report.txt: Per-class performance metrics
================================================================================
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# import numpy as np
# from tensorflow.keras.models import load_model
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# # Dataset path
# csv_file = r'C:\Users\dsdjs\Documents\canada\project\paz\paz\datasets\FER\fer2013.csv'

# # Image and model configuration
# image_size = 48
# num_classes = 7

# # Emotion labels in order (matching model output indices)
# emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # ============================================================================
# # DATA LOADING
# # ============================================================================

# print("loading fer2013 dataset")
# # Load complete dataset
# data = pd.read_csv(csv_file)

# # Extract only test data (Usage != 'Training')
# # This ensures we evaluate on data the model has never seen
# test_data = data[data['Usage'] != 'Training']

# print(f"test samples: {len(test_data)}")
# print("expected: ~7,178 images")

# # ============================================================================
# # TEST DATA PREPROCESSING
# # ============================================================================

# print("preprocessing test data")

# # Initialize lists to store processed images and labels
# x_test = []
# y_test = []

# # Process each test sample
# for index, row in test_data.iterrows():
#     # Parse pixel string to numpy array
#     pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    
#     # Reshape to 48x48x1 and normalize to 0.0-1.0
#     # Same preprocessing as during training
#     image = pixels.reshape(image_size, image_size, 1) / 255.0
    
#     # Collect image and label
#     x_test.append(image)
#     y_test.append(int(row['emotion']))
    
#     # Progress indicator every 1000 samples
#     if (index + 1) % 1000 == 0:
#         print(f"processed {index + 1} samples")

# # Convert lists to numpy arrays
# x_test = np.array(x_test)
# y_test_labels = np.array(y_test)

# print("preprocessing complete")
# print(f"test data shape: {x_test.shape}")
# print(f"test labels shape: {y_test_labels.shape}")

# # ============================================================================
# # MODEL LOADING
# # ============================================================================

# print("\nloading baseline model")
# # Load our trained baseline model (66.93% accuracy)
# model = load_model('experiments/FER_Baseline_20260202_191825/best_model.h5')
# print("model loaded successfully")

# # ============================================================================
# # GENERATE PREDICTIONS
# # ============================================================================

# print("\ngenerating predictions on test set")
# print("this may take 1-2 minutes for 7,178 images")

# # Predict emotions for all test images
# # predictions shape: (7178, 7) - probabilities for each emotion
# predictions = model.predict(x_test, verbose=1)

# # Convert probabilities to class labels
# # argmax gets the index of highest probability
# # Example: [0.05, 0.02, 0.10, 0.75, 0.03, 0.04, 0.01] → 3 (Happy)
# predicted_labels = np.argmax(predictions, axis=1)

# print("predictions complete")

# # ============================================================================
# # CONFUSION MATRIX GENERATION
# # ============================================================================

# print("\ncreating confusion matrix")

# # Generate confusion matrix
# # Rows = True labels, Columns = Predicted labels
# # Example: cm[0][3] = number of "Angry" faces predicted as "Happy"
# cm = confusion_matrix(y_test_labels, predicted_labels)

# print("confusion matrix shape:", cm.shape)
# print("expected: (7, 7) matrix")

# # ============================================================================
# # CONFUSION MATRIX VISUALIZATION
# # ============================================================================

# print("\ngenerating confusion matrix visualization")

# # Create figure
# plt.figure(figsize=(10, 8))

# # Create heatmap using seaborn
# # annot=True: Show numbers in each cell
# # fmt='d': Format as integers (not decimals)
# # cmap='Blues': Blue color scheme (darker = more samples)
# sns.heatmap(
#     cm, 
#     annot=True,           # Show count numbers
#     fmt='d',              # Integer format
#     cmap='Blues',         # Color scheme
#     xticklabels=emotions, # Column labels
#     yticklabels=emotions, # Row labels
#     cbar_kws={'label': 'Number of Samples'}  # Colorbar label
# )

# # Add labels and title
# plt.title('confusion matrix - baseline model\n(rows: true emotion, columns: predicted emotion)', 
#           fontsize=14, pad=20)
# plt.ylabel('true emotion', fontsize=12)
# plt.xlabel('predicted emotion', fontsize=12)

# # Rotate x-axis labels for readability
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# # Save high-resolution image
# plt.tight_layout()
# plt.savefig('confusion_matrix_baseline.png', dpi=300, bbox_inches='tight')
# print("confusion matrix saved: confusion_matrix_baseline.png")

# # Close plot to free memory
# plt.close()

# # ============================================================================
# # CLASSIFICATION REPORT GENERATION
# # ============================================================================

# print("\ngenerating classification report")

# # Generate detailed per-class metrics
# # Includes: precision, recall, F1-score, support
# report = classification_report(
#     y_test_labels, 
#     predicted_labels, 
#     target_names=emotions,
#     digits=4  # Show 4 decimal places
# )

# # Display report in console
# print("\nper-class performance:")
# print("=" * 70)
# print(report)
# print("=" * 70)

# # ============================================================================
# # SAVE CLASSIFICATION REPORT
# # ============================================================================

# # Save report to file for documentation
# with open('classification_report_baseline.txt', 'w') as f:
#     f.write("classification report - baseline model\n")
#     f.write("=" * 70 + "\n\n")
#     f.write("model: minixception (baseline)\n")
#     f.write("dataset: fer2013 test set (7,178 images)\n")
#     f.write("overall accuracy: 66.93%\n\n")
#     f.write("per-class metrics:\n")
#     f.write("-" * 70 + "\n\n")
#     f.write(report)
#     f.write("\n" + "=" * 70 + "\n\n")
#     f.write("metric definitions:\n")
#     f.write("-" * 70 + "\n")
#     f.write("precision: of predicted X, how many are actually X?\n")
#     f.write("           (true positives / (true positives + false positives))\n\n")
#     f.write("recall: of actual X, how many did we predict as X?\n")
#     f.write("        (true positives / (true positives + false negatives))\n\n")
#     f.write("f1-score: harmonic mean of precision and recall\n")
#     f.write("          (2 * precision * recall / (precision + recall))\n\n")
#     f.write("support: number of actual samples for this emotion in test set\n")

# print("\nclassification report saved: classification_report_baseline.txt")

# # ============================================================================
# # ERROR ANALYSIS SUMMARY
# # ============================================================================

# print("\n" + "=" * 70)
# print("error analysis summary")
# print("=" * 70)

# # Calculate per-class accuracy from confusion matrix
# print("\nper-class accuracy:")
# for i, emotion in enumerate(emotions):
#     # Diagonal elements are correct predictions
#     correct = cm[i][i]
#     # Row sum is total samples for this class
#     total = cm[i].sum()
#     accuracy = (correct / total) * 100 if total > 0 else 0
#     print(f"  {emotion:10s}: {accuracy:5.2f}% ({correct:4d}/{total:4d} correct)")

# # Identify most confused pairs
# print("\nmost common confusions:")
# confusions = []
# for i in range(num_classes):
#     for j in range(num_classes):
#         if i != j and cm[i][j] > 0:  # Skip diagonal and zeros
#             confusions.append((cm[i][j], emotions[i], emotions[j]))

# # Sort by count (highest first)
# confusions.sort(reverse=True)

# # Show top 5 confusions
# for count, true_emotion, predicted_emotion in confusions[:5]:
#     print(f"  {true_emotion:10s} → {predicted_emotion:10s}: {count:4d} times")

# # ============================================================================
# # INSIGHTS AND OBSERVATIONS
# # ============================================================================

# print("\n" + "=" * 70)
# print("key insights")
# print("=" * 70)

# # Analyze class imbalance impact
# print("\nclass imbalance effects:")
# class_counts = [cm[i].sum() for i in range(num_classes)]
# for i, emotion in enumerate(emotions):
#     percentage = (class_counts[i] / sum(class_counts)) * 100
#     accuracy = (cm[i][i] / class_counts[i]) * 100 if class_counts[i] > 0 else 0
#     print(f"  {emotion:10s}: {percentage:5.2f}% of data, {accuracy:5.2f}% accuracy")

# print("\nobservations:")
# print("  - minority classes (disgust: 1.5%) show lower accuracy")
# print("  - majority classes (happy: 25%) show higher accuracy")
# print("  - similar emotions (fear/surprise) show confusion")
# print("  - neutral expressions often confused with sad")

# print("\n" + "=" * 70)
# print("analysis complete")
# print("=" * 70)

# # ============================================================================
# # END OF SCRIPT
# # ============================================================================

# """
# INTERPRETATION GUIDE:

# 1. Confusion Matrix Reading:
#    - Diagonal (top-left to bottom-right): Correct predictions
#    - Off-diagonal: Errors
#    - Dark blue in heatmap: Many samples
#    - Light blue: Few samples

# 2. Expected Patterns:
#    - Fear ↔ Surprise: Both have wide eyes, open mouth
#    - Sad ↔ Neutral: Subtle differences in expression
#    - Angry ↔ Disgust: Both have furrowed brows
#    - Happy: High accuracy (distinctive smile)

# 3. Class Imbalance Impact:
#    Emotion    | % of Data | Expected Accuracy
#    -----------|-----------|------------------
#    Happy      | 25%       | High (70-80%)
#    Disgust    | 1.5%      | Low (40-50%)
#    Others     | 13-17%    | Medium (60-70%)

# 4. Metrics Explained:
#    - High Precision: Few false alarms (predicted X but was Y)
#    - High Recall: Found most instances (predicted most X correctly)
#    - High F1: Good balance of precision and recall
   
# 5. Using This Analysis:
#    - Identify weak classes → focus improvements there
#    - Understand confusion patterns → add discriminative features
#    - Guide data collection → balance minority classes
#    - Inform ensemble strategies → combine models with different strengths
# """