import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Charger les données
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Normaliser les longueurs des données
max_length = max(len(seq) for seq in data)
data = [seq + [0] * (max_length - len(seq)) for seq in data]  # Compléter avec des zéros
data = np.asarray(data)
labels = np.asarray(labels)

# Séparation des données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Modèle de classification
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Prédictions et score
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% des échantillons ont été classifiés correctement !')

# Sauvegarde du modèle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
