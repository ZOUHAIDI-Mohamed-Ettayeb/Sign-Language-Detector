import os
import pickle
import cv2
import mediapipe as mp

# Initialisation de Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Définition du répertoire de données
DATA_DIR = './data'

# Vérifie si le dossier existe, sinon crée-le
if not os.path.exists(DATA_DIR):
    print(f"Le dossier {DATA_DIR} n'existe pas, création du dossier.")
    os.makedirs(DATA_DIR)

# Listes pour stocker les données et les étiquettes
data = []
labels = []

# Parcours des sous-dossiers et images dans DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):  # S'assurer que c'est un dossier
        continue

    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)

        try:
            img = cv2.imread(img_full_path)
            if img is None:  # Vérifie si l'image a été lue correctement
                print(f"Erreur lors de la lecture de l'image: {img_full_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalisation des coordonnées
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Vérifiez que les données ont une longueur attendue (42 pour une seule main)
                    expected_length = 42  # 21 points x 2 (x, y)
                    if len(data_aux) == expected_length:
                        data.append(data_aux)
                        labels.append(dir_.replace('Label-', ''))
                    else:
                        print(f"Données ignorées pour l'image {img_full_path} en raison d'une longueur incohérente.")

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {img_full_path}: {e}")
            continue

# Sauvegarde des données et des labels dans un fichier pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Données sauvegardées dans 'data.pickle' avec {len(data)} exemples.")
