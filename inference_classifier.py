import pickle
import cv2
import mediapipe as mp
import numpy as np

# Charger le modèle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialisation de la reconnaissance des mains
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionnaire des labels
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'W'}  # Adapte ce dictionnaire selon tes labels

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    # Convertir l'image en RGB pour le traitement de MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processer l'image pour détecter les mains
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les repères des mains
            mp_drawing.draw_landmarks(
                frame,  # image
                hand_landmarks,  # landmarks
                mp_hands.HAND_CONNECTIONS,  # connexions
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Sauvegarder les coordonnées x et y normalisées
                x_.append(x)
                y_.append(y)

            # Calculer la différence des coordonnées pour chaque landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculer les coordonnées de la zone de détection
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Faire la prédiction
        prediction = model.predict([np.asarray(data_aux)])

        # Vérification du format de la prédiction (si c'est un label comme 'Lettre-W')
        predicted_value = prediction[0]

        if isinstance(predicted_value, str) and '-' in predicted_value:
            # Si la prédiction est sous forme 'Lettre-X', on extrait la lettre
            predicted_character = predicted_value.split('-')[1]
        else:
            # Si c'est un entier, on l'utilise comme index pour le dictionnaire
            predicted_character = labels_dict.get(int(predicted_value), 'Inconnu')

        # Afficher un rectangle autour de la main et la lettre prédite
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Afficher la vidéo avec la prédiction
    cv2.imshow('frame', frame)

    # Quitter lorsque la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
