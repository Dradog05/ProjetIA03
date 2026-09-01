"""
Machine B - Reception du flux de la machine A + detection de personnes (YOLOv8).

Prerequis : pip install -r requirements.txt
Le modele yolov8n.pt (~6 Mo) est telecharge automatiquement au premier lancement.

Utilisation :
    python person_detection.py --stream-url http://<IP_MACHINE_A>:5000/video_feed

Appuie sur 'q' dans la fenetre video pour quitter.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # classe "person" dans le dataset COCO (utilise par YOLOv8 pre-entraine)


def main(stream_url: str, model_name: str, conf_threshold: float):
    print("Chargement du modele YOLOv8...")
    model = YOLO(model_name)

    print(f"Connexion au flux : {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(
            "Impossible d'ouvrir le flux. Verifie l'IP/le port et que les deux "
            "machines sont bien sur le meme reseau."
        )

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Flux interrompu, tentative de reconnexion...")
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)
            continue

        results = model(frame, classes=[PERSON_CLASS_ID], conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        nb_personnes = len(results[0].boxes)
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        cv2.putText(
            annotated_frame,
            f"Personnes detectees: {nb_personnes} | FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Detection de personnes - Machine B", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream-url", required=True, help="URL du flux MJPEG de la machine A"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt", help="Modele YOLOv8 a utiliser (nano par defaut)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Seuil de confiance (0 a 1)"
    )
    args = parser.parse_args()

    main(args.stream_url, args.model, args.conf)
