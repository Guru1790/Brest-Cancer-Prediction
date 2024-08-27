#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


data = pd.read_csv("C:\\Users\\91762\\Downloads\\data.csv", index_col=False)
data.drop('Unnamed: 32', axis=1, inplace=True)


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


# Convert diagnosis column to numerical values
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Calculate correlation between features and diagnosis
feature_diagnosis_corr = data.corr()['diagnosis'].sort_values(ascending=False)

# Plot clustered bar plot
plt.figure(figsize=(14, 8))
feature_diagnosis_corr.drop('diagnosis').plot(kind='bar', color=['skyblue' if x < 0 else 'salmon' for x in feature_diagnosis_corr.drop('diagnosis')], edgecolor='black')
plt.title('Feature Correlation with Diagnosis', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# # Splitting the Data

# In[7]:


# Assign predictors to a variable of ndarray (matrix) type
array = data.values
X = array[:, 2:]  # Exclude 'id' column
y = array[:, 1]   # 'diagnosis' column


# In[8]:


# Transform the class labels from their original string representation (M and B) into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[9]:


# Split your data
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[10]:


# Normalize the data (center around 0 and scale to remove the variance)
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)


# In[11]:


# Function to train and evaluate a classifier
def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')


# # Training The Data

# In[12]:


# SVM
svm_model = SVC(kernel='linear', C=1, probability=True)
print("Support Vector Machine:")
train_and_evaluate_classifier(svm_model, Xs_train, y_train_encoded, Xs_test, y_test_encoded)
print("\n")


# In[13]:


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Random Forest:")
train_and_evaluate_classifier(rf_model, Xs_train, y_train_encoded, Xs_test, y_test_encoded)
print("\n")


# In[14]:


# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
print("K-Nearest Neighbors:")
train_and_evaluate_classifier(knn_model, Xs_train, y_train_encoded, Xs_test, y_test_encoded)
print("\n")


# In[15]:


# Logistic Regression
lr_model = LogisticRegression(random_state=42)
print("Logistic Regression:")
train_and_evaluate_classifier(lr_model, Xs_train, y_train_encoded, Xs_test, y_test_encoded)


# In[16]:


# Assuming you have the following models and accuracies defined earlier in your code
models = ['SVM', 'Random Forest', 'K-Nearest Neighbors', 'Logistic Regression']
accuracies = [0.9561, 0.9649, 0.9474, 0.9737]

# Now you can use these variables to create the bar chart
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0.9, 1.0)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()


# In[17]:


# ROC Curve for Random Forest
y_prob_lr = lr_model.predict_proba(Xs_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_prob_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# # Predicting using Logistic Regression

# In[18]:


# Define the input data
input_data = np.array([11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563]).reshape(1, -1)

# Make predictions using the Logistic Regression
prediction = lr_model.predict(input_data)
print(prediction)

# Interpret the prediction
if prediction[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')


# In[ ]:




