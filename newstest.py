# Fake News Detection - Visual Analysis and Evaluation

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels: 1 for fake, 0 for true
df_fake['label'] = 1
df_true['label'] = 0

# Combine and clean
df_combined = pd.concat([df_fake, df_true], ignore_index=True)
df_combined['text'] = df_combined['title'] + " " + df_combined['text']
df_combined = df_combined.fillna("")

# Generate word clouds for fake and real news
fake_text = " ".join(df_combined[df_combined['label'] == 1]['text'])
true_text = " ".join(df_combined[df_combined['label'] == 0]['text'])

wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
wordcloud_true = WordCloud(width=800, height=400, background_color='white').generate(true_text)

# Plot word clouds
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title("Fake News Word Cloud")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_true, interpolation='bilinear')
plt.title("Real News Word Cloud")
plt.axis('off')

plt.tight_layout()
plt.show()

### 🚧 Add your model's predictions
# Before running the next section, define these variables:
# y_true: Ground truth labels
# y_pred: Predicted labels from your model
# y_probs: Predicted probabilities (used for ROC curve)

# Example (to be replaced with actual predictions):
# y_true = [0, 1, 0, 1, ...]
# y_pred = [0, 1, 1, 0, ...]
# y_probs = [0.2, 0.8, 0.6, 0.4, ...]

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
