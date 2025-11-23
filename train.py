import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Criar a pasta models se não existir
if not os.path.exists('models'):
    os.makedirs('models')

print("Treinamento iniciado...")

# Carregar dados
iris = load_iris()
X = iris.data
y = iris.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avaliar
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

# Salvar modelo
model_path = 'models/model.pkl'
joblib.dump(clf, model_path)
print(f"Modelo salvo em {model_path}")
