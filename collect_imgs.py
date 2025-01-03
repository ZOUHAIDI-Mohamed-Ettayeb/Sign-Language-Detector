import os
import cv2
import string

# Chemin vers le répertoire des données
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Liste des lettres (modifiez si nécessaire)
lettres = list(string.ascii_uppercase)  # ['A', 'B', 'C', ..., 'Z']

# Taille du dataset pour chaque lettre
dataset_size = 100

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Collecte des images pour chaque lettre
for index, lettre in enumerate(lettres):
    dossier_lettre = f'Lettre-{lettre}'  # Nom du dossier pour la lettre
    chemin_dossier = os.path.join(DATA_DIR, dossier_lettre)
    
    if not os.path.exists(chemin_dossier):
        os.makedirs(chemin_dossier)

    print(f'Collecte des données pour la classe {lettre} (index {index})')

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
