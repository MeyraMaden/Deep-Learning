import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#giriş
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

#beklenen çıktı
y = np.array([0, 1, 1, 0])


#çok katmanlı yapay sinir ağı
model = MLPClassifier(
    hidden_layer_sizes=(4,),  #gizli katmanda 4 nöron
    activation='logistic',        #aktivasyon fonk -> sigmoid
    solver='lbfgs',             # küçük veri setlerinde öğrenmenin daha stabil olması için
    max_iter=1000,            
    random_state=42
)

model.fit(X, y)
predictions = model.predict(X)

print("\n---Model Sonuçları---")
print("Doğruluk:", accuracy_score(y, predictions))
for i in range(len(X)):
    print(
        "Girdi:", X[i],
        "Gerçek:", y[i],
        "Tahmin:", predictions[i]
    )
