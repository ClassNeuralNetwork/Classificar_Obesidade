import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE


df = pd.read_csv('./dataset/ObesityDataSet.csv')

print(df.head())
print(df.info())

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
df['MTRANS'] = df['MTRANS'].map({'Walking': 0, 'Motorbike': 1, 'Automobile': 2, 'Public_Transportation': 3})

df['NObeyesdad'] = df['NObeyesdad'].astype('category').cat.codes

print(df.info())

print(df['NObeyesdad'].value_counts())


