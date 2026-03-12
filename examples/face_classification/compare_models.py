import pandas as pd
import matplotlib.pyplot as plt



#Comprehensive comparison of all three models

results = {
    'Model': ['Baseline', 'Attention', 'Histogram'],
    'Test Accuracy': [66.93, 58.71, 62.47],
    'Parameters': ['3.9M', '4.2M', '3.9M'],
    'Training Time': ['45 min', '35 min', '35 min']
}


# Convert to DataFrame for easy manipulation
df = pd.DataFrame(results)
df.to_csv('model_comparison.csv', index=False)

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(10, 6))
models = df['Model']
accuracies = df['Test Accuracy']

# Create bar chart with distinct colors for each model
bars = ax.bar(models, accuracies, color=['blue', 'green', 'orange'])
ax.set_ylabel('test accuracy (%)')
ax.set_title('model performa comparison')
ax.set_ylim([55, 75])

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.2f}%',
           ha='center', va='bottom')

# Save high-resolution figure
plt.tight_layout()
plt.savefig('model.png', dpi=300)

plt.close()









# code 2

"""
================================================================================
COSC-4427: Computer Vision Project
Model Performance Comparison - Results Analysis

Author: Student 2 (Model Architecture Lead)
Date: February 2026
Course: Computer Vision (Winter 2026)
Professor: Omar Al-Buraiki

Description:
This script compares the performance of all three models developed in this
project: baseline, attention-enhanced, and histogram-equalized. It generates
a comprehensive comparison table and visualization showing which approach
worked best for facial emotion recognition.

Models Compared:
1. Baseline: MiniXception with standard preprocessing (66.93%)
2. Attention: Custom CNN with SE blocks (Student 2's contribution)
3. Histogram: MiniXception with histogram equalization (Student 1's contribution)

Purpose:
- Evaluate effectiveness of our modifications
- Demonstrate systematic experimentation
- Guide conclusions and future work
- Create presentation-ready visualizations

Output Files:
- model_comparison.csv: Tabular results data
- model_comparison.png: Bar chart visualization
================================================================================
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import pandas as pd
# import matplotlib.pyplot as plt

# # ============================================================================
# # EXPERIMENTAL RESULTS
# # ============================================================================

# # Comprehensive comparison of all three models
# # Results collected from experiment_summary.txt files
# results = {
#     'Model': ['Baseline', 'Attention', 'Histogram'],
    
#     # Test accuracy on FER2013 test set (7,178 images)
#     # Higher is better
#     'Test Accuracy': [66.93, 58.71, 62.47],
    
#     # Model size (total parameters)
#     # Attention model larger due to SE blocks
#     'Parameters': ['3.9M', '4.2M', '3.9M'],
    
#     # Training time (approximate)
#     # Baseline took longer due to 45 epochs vs 30 epochs for others
#     'Training Time': ['45 min', '35 min', '35 min']
# }

# # ============================================================================
# # RESULTS ANALYSIS
# # ============================================================================

# print("=" * 70)
# print("model comparison - experimental results")
# print("=" * 70)

# # Convert to DataFrame for easy manipulation
# df = pd.DataFrame(results)

# # Display results table
# print("\nperformance summary:")
# print("-" * 70)
# print(df.to_string(index=False))
# print("-" * 70)

# # ============================================================================
# # STATISTICAL ANALYSIS
# # ============================================================================

# print("\nanalysis:")

# # Find best model
# best_idx = df['Test Accuracy'].idxmax()
# best_model = df.loc[best_idx, 'Model']
# best_accuracy = df.loc[best_idx, 'Test Accuracy']

# print(f"  best performing model: {best_model} ({best_accuracy:.2f}%)")

# # Calculate improvement over baseline
# baseline_accuracy = df[df['Model'] == 'Baseline']['Test Accuracy'].values[0]
# for idx, row in df.iterrows():
#     if row['Model'] != 'Baseline':
#         diff = row['Test Accuracy'] - baseline_accuracy
#         direction = "improvement" if diff > 0 else "decrease"
#         print(f"  {row['Model']:10s} vs baseline: {abs(diff):+5.2f}% {direction}")

# # ============================================================================
# # SAVE RESULTS TABLE
# # ============================================================================

# # Save to CSV for easy reference and documentation
# df.to_csv('model_comparison.csv', index=False)
# print("\nresults saved: model_comparison.csv")

# # ============================================================================
# # VISUALIZATION - BAR CHART
# # ============================================================================

# print("\ngenerating visualization")

# # Create figure with appropriate size
# fig, ax = plt.subplots(figsize=(10, 6))

# # Extract data for plotting
# models = df['Model']
# accuracies = df['Test Accuracy']

# # Create bar chart with distinct colors for each model
# bars = ax.bar(
#     models, 
#     accuracies, 
#     color=['blue', 'green', 'orange'],
#     alpha=0.8,
#     edgecolor='black',
#     linewidth=1.5
# )

# # Customize appearance
# ax.set_ylabel('test accuracy (%)', fontsize=12, fontweight='bold')
# ax.set_xlabel('model variant', fontsize=12, fontweight='bold')
# ax.set_title('model performance comparison\nfer2013 emotion recognition', 
#              fontsize=14, fontweight='bold', pad=20)

# # Set y-axis range to highlight differences
# # Range: 55-75% shows variation clearly
# ax.set_ylim([55, 75])

# # Add grid for easier reading
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on top of each bar
# for bar in bars:
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width() / 2.,  # X position (center of bar)
#         height,                               # Y position (top of bar)
#         f'{height:.2f}%',                    # Label text
#         ha='center',                          # Horizontal alignment
#         va='bottom',                          # Vertical alignment
#         fontsize=11,
#         fontweight='bold'
#     )

# # Add legend explaining each model
# legend_labels = [
#     'Baseline: MiniXception + standard preprocessing',
#     'Attention: Custom CNN + SE blocks (Student 2)',
#     'Histogram: MiniXception + histogram eq (Student 1)'
# ]
# ax.legend(bars, legend_labels, loc='upper right', fontsize=9)

# # Save high-resolution figure
# plt.tight_layout()
# plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
# print("visualization saved: model_comparison.png")

# # Close plot to free memory
# plt.close()

