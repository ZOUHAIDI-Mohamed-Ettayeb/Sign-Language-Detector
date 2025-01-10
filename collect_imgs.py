import os
import cv2
import string

# Chemin vers le répertoire des données
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Liste des labels (lettres et mots supplémentaires)
labels = list(string.ascii_uppercase) + ['hello', 'thanks', 'yes', 'no', 'i_love_you']

# Taille du dataset pour chaque label
dataset_size = 100

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Collecte des images pour chaque label
for index, label in enumerate(labels):
    dossier_label = f'Label-{label}'  # Nom du dossier pour le label
    chemin_dossier = os.path.join(DATA_DIR, dossier_label)
    
    if not os.path.exists(chemin_dossier):
        os.makedirs(chemin_dossier)

    print(f'Collecte des données pour la classe {label} (index {index})')

    # Préparation avant la collecte
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pret? Appuyez sur "Q" pour commencer!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collecte des images
    compteur = 0
    while compteur < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la caméra.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Sauvegarde de l'image
        chemin_image = os.path.join(chemin_dossier, f'{compteur}.jpg')
        cv2.imwrite(chemin_image, frame)
        compteur += 1

cap.release()
cv2.destroyAllWindows()
