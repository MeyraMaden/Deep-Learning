import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# veri yükleme
df = pd.read_csv("data.csv")

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

# gereksiz sütunları sil
if 'id' in df.columns:
    df = df.drop(columns=['id'])

if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# train-test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardizasyon 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# modeli eğitme 
model = LogisticRegression()
model.fit(X_train, y_train)
#tahmin 
y_pred = model.predict(X_test)

# performans metriklerinin hesaplanması
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(confusion_matrix(y_test, y_pred))

