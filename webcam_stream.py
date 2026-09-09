"""
Machine A - Diffusion du flux webcam en streaming MJPEG sur le reseau local.

Utilisation :
    python webcam_stream.py [--port 5000] [--camera 0]

Une fois lance, le flux est accessible depuis n'importe quelle machine du
meme reseau local a l'adresse : http://<IP_DE_CETTE_MACHINE>:<port>/video_feed
"""

import argparse
import socket

import cv2
from flask import Flask, Response

#définition de l'application Flask
app = Flask(__name__)

def get_local_ip():
    """Recupere l'adresse IP locale de la machine sur le reseau (pour affichage)."""
    # Connexion UDP temporaire pour connaitre l'IP locale utilisee.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        # Utilise localhost si aucune IP reseau n'est disponible.
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_frames(camera_index: int):
    """Produit le flux d'images JPEG qui sera par la suite transmis aux autres appareils"""
    # Ouverture de la webcam.
    camera = cv2.VideoCapture(camera_index)

    # Vérification de l'ouverture de la caméra
    if not camera.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera {camera_index}")

    try:
        while True:
            # Capture d'une image depuis la webcam.
            success, frame = camera.read()
            if not success:
                break

            # Conversion de l'image en JPEG pour le streaming.
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60]) #la qualité de l'image est fixée à 60 pour réduire la taille du flux et améliorer la fluidité.
            if not ok:
                continue

            # Envoie l'image au format MJPEG.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        camera.release()


@app.route("/video_feed")   #Si quelqu'un demande l'adresse ip+/video_feed : on appelle la fonction video_feed()
def video_feed():
    """Retourne le flux MJPEG (JPG en temps que flux) de la webcam."""
    
    return Response(
        generate_frames(app.config["CAMERA_INDEX"]),
        mimetype="multipart/x-mixed-replace; boundary=frame", #indique que le flux est de type multipart et que chaque image est séparée par la chaîne "frame"
    ) 


@app.route("/")
def index():
    # Page d'accueil avec affichage du flux vidéo.
    return "<h1>Flux webcam actif</h1><img src='/video_feed' width='720'>"


if __name__ == "__main__":
    # Definition des arguments utilisables depuis le terminal.
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument( "--camera", type=int, default=0, help="Index de la webcam (0 par defaut)")
    args = parser.parse_args()

    # Stocke l'index de la webcam dans la configuration Flask.
    app.config["CAMERA_INDEX"] = args.camera

    # Affiche l'adresse du flux dans le terminal.
    local_ip = get_local_ip()
    print(f"Flux disponible sur : http://{local_ip}:{args.port}/video_feed")

    # Lance le serveur Flask et autorise les connexions du reseau local.
    app.run(host="0.0.0.0", port=args.port, threaded=True)
