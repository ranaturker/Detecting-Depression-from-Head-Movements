import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Veriyi işleme
data = """
Index,Accuracy
1,82.98
2,74.47
3,54.61
4,48.23
5,66.67
6,71.63
7,70.21
8,84.40
9,76.60
10,29.08
11,54.61
12,70.92
13,63.83
14,69.50
15,39.01
16,64.54
17,28.37
18,26.95
19,81.56
20,71.63
21,76.60
22,46.10
23,57.45
24,79.43
25,43.97
26,80.14
27,58.87
28,70.92
29,85.82
30,33.33
31,61.70
32,77.30
33,70.92
34,51.77
35,70.21
36,79.43
37,82.98
38,52.48
39,70.92
40,43.97
41,87.94
42,39.72
43,66.67
44,76.60
45,85.82
46,35.46
47,73.05
48,57.45
49,70.21
50,56.03
51,79.43
52,64.54
53,70.21
54,75.89
55,73.05
56,72.34
57,85.82
58,51.77
59,29.08
60,77.30
61,46.81
62,45.39
63,74.47
64,92.20
65,78.72
66,51.06
67,38.30
68,73.05
69,75.89
70,59.57
71,69.50
72,21.28
73,27.66
74,75.89
75,75.18
76,69.50
77,87.94
78,62.41
79,32.62
80,87.23
81,77.30
82,67.38
83,51.06
84,69.50
85,68.09
86,70.92
87,41.13
88,75.18
89,36.17
90,67.38
91,60.99
92,70.21
93,74.47
94,62.41
95,34.04
96,31.21
97,28.37
98,80.85
99,54.61
100,44.68
"""

# Veriyi DataFrame'e çevirme
from io import StringIO
df = pd.read_csv(StringIO(data))

# Histogram oluşturma
plt.figure(figsize=(12, 6))
plt.hist(df['Accuracy'], bins=20, alpha=0.7, edgecolor='black',color='saddlebrown')

plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracy Across Indices')
plt.grid(True)
plt.show()

# En yüksek ve en düşük doğruluk değerlerini belirleme
highest = df.loc[df['Accuracy'].idxmax()]
lowest = df.loc[df['Accuracy'].idxmin()]



# Genel istatistikleri hesaplama
stats = df['Accuracy'].describe()
stats


import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Örnek gerçek ve tahmin edilen değerler (örnek veriler)
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)  # Gerçek sınıflar
y_pred = np.random.randint(0, 2, 100)  # Tahmin edilen sınıflar

# Karışıklık matrisi hesaplama
cm = confusion_matrix(y_true, y_pred)

# Görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Confusion matrix values based on the classification report
cm = np.array([[100, 0], [6, 35]])

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Index228_combined_matrix.csv')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# True labels and predicted probabilities for positive class (1)
y_true = [0] * 100 + [1] * 41  # Based on support values
y_scores = [0.94] * 100 + [1.00] * 35 + [0.85] * 6  # Approximation from classification report

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve and area
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plot ROC Curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Index numaraları ve doğruluk oranları
indices = np.arange(1, 289)
accuracies = [
    82.98, 74.47, 54.61, 48.23, 66.67, 71.63, 70.21, 84.40, 76.60, 29.08,
    54.61, 70.92, 63.83, 69.50, 39.01, 64.54, 28.37, 26.95, 81.56, 71.63,
    76.60, 46.10, 57.45, 79.43, 43.97, 80.14, 58.87, 70.92, 85.82, 33.33,
    61.70, 77.30, 70.92, 51.77, 70.21, 79.43, 82.98, 52.48, 70.92, 43.97,
    87.94, 39.72, 66.67, 76.60, 85.82, 35.46, 73.05, 57.45, 70.21, 56.03,
    79.43, 64.54, 70.21, 75.89, 73.05, 72.34, 85.82, 51.77, 29.08, 77.30,
    46.81, 45.39, 74.47, 92.20, 78.72, 51.06, 38.30, 73.05, 75.89, 59.57,
    69.50, 21.28, 27.66, 75.89, 75.18, 69.50, 87.94, 62.41, 32.62, 87.23,
    77.30, 67.38, 51.06, 69.50, 68.09, 70.92, 41.13, 75.18, 36.17, 67.38,
    60.99, 70.21, 74.47, 62.41, 34.04, 31.21, 28.37, 80.85, 54.61, 44.68,
    78.01, 63.83, 73.76, 62.41, 60.28, 66.67, 15.60, 42.55, 45.39, 70.92,
    53.90, 70.92, 56.03, 57.45, 71.63, 69.50, 60.99, 80.85, 70.21, 36.88,
    42.55, 46.81, 65.96, 57.45, 76.60, 82.98, 74.47, 68.79, 78.72, 72.34,
    53.90, 73.76, 70.92, 63.12, 70.92, 85.11, 72.34, 74.47, 75.89, 81.56,
    63.12, 75.18, 70.92, 36.17, 18.44, 68.09, 69.50, 68.09, 70.92, 49.65,
    47.52, 79.43, 67.38, 32.62, 62.41, 66.67, 69.50, 70.21, 19.86, 71.63,
    58.16, 68.79, 70.92, 43.97, 66.67, 65.96, 85.11, 46.81, 46.10, 78.72,
    43.26, 65.25, 56.74, 60.28, 63.12, 77.30, 60.99, 83.69, 67.38, 70.21,
    80.14, 66.67, 60.28, 53.90, 67.38, 71.63, 63.12, 25.53, 34.04, 72.34,
    29.08, 71.63, 64.54, 80.14, 79.43, 67.38, 31.91, 73.76, 68.79, 29.08,
    70.92, 83.69, 85.82, 68.79, 71.63, 78.72, 67.38, 21.99, 65.25, 56.03,
    32.62, 68.09, 73.76, 80.14, 68.79, 67.38, 88.65, 77.30, 78.01, 78.72,
    73.76, 72.34, 71.63, 68.09, 58.87, 77.30, 69.50, 95.74, 68.79, 73.05,
    31.91, 29.08, 27.66, 62.41, 48.23, 69.50, 77.30, 78.01, 43.26, 78.01,
    71.63, 47.52, 69.50, 70.92, 70.92, 51.77, 69.50, 51.77, 70.92, 32.62,
    70.92, 76.60, 73.76, 72.34, 32.62, 70.92, 47.52, 24.11, 73.76, 52.48,
    77.30, 39.01, 34.04, 82.98, 66.67, 43.26, 33.33, 66.67, 58.16, 42.55,
    65.96, 68.09, 56.03, 71.63, 70.92, 72.34, 85.82, 36.88, 73.76, 63.12,
    70.92, 64.54, 70.92, 63.12, 68.79, 70.92, 79.43, 60.28
]

# Çizgi grafiği çizimi

plt.figure(figsize=(15, 6))
plt.plot(indices, accuracies, marker='o', linestyle='-', markersize=3)
plt.title('Accuracy Values Across 288 Indices')
plt.xlabel('Index Number')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# X ekseni etiketlerinin her birini tek tek yazdır, font boyutunu 3 yap
plt.xticks(indices, rotation=90, fontsize=4)
plt.tight_layout()
plt.show()
