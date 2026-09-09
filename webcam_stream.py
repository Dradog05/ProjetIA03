"""
Machine A - Diffusion du flux webcam en streaming MJPEG sur le reseau local.

Utilisation :
    python webcam_stream.py [--port 5000] [--camera 0]

Une fois lance, le flux est accessible depuis n'importe quelle machine du
meme reseau local a l'adresse : http://<IP_DE_CETTE_MACHINE>:<port>/video_feed
On peut aussi l'ouvrir directement dans un navigateur pour verifier que ca marche.
"""

import argparse   # lecture des options passees en ligne de commande (--port, --camera)
import socket     # utilise dans get_local_ip() pour trouver l'adresse IP de la machine

import cv2                              # OpenCV : acces a la webcam, encodage JPEG
from flask import Flask, Response       # mini-serveur web : gere le protocole HTTP a notre place

# Cree l'application web. Il ne reste plus qu'a definir une fonction par
# URL ("route") pour dire quoi repondre a chaque demande.
app = Flask(__name__)


def get_local_ip():
    """Recupere l'adresse IP locale de la machine sur le reseau (pour affichage)."""
    # Astuce : on "fait semblant" de se connecter a une adresse externe
    # (8.8.8.8, un serveur DNS public de Google) juste pour demander au
    # systeme d'exploitation quelle IP locale serait utilisee pour cette
    # connexion. Aucune donnee n'est reellement envoyee sur le reseau.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"  # repli si la machine n'a pas de reseau du tout
    finally:
        s.close()
    return ip


def generate_frames(camera_index: int):
    # Ouvre la webcam. C'est seulement a cet instant (premier appel de la
    # fonction, donc premiere connexion d'un client a /video_feed) que la
    # camera s'allume reellement - pas au demarrage du script.
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera {camera_index}")

    try:
        while True:
            # Capture une image (une "frame") a l'instant present.
            success, frame = camera.read()
            if not success:
                break

            # Compresse l'image en JPEG, en memoire (pas de fichier sur le
            # disque). Qualite 80/100 : bon compromis poids/qualite/vitesse.
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            # Format standard MJPEG : chaque image est precedee d'un
            # marqueur "--frame" et d'un en-tete Content-Type. Le navigateur
            # (ou OpenCV sur une autre machine) sait lire ce format nativement
            # et l'affiche comme une video continue. Le mot-cle `yield` (au
            # lieu de `return`) permet de renvoyer une image a la fois, sans
            # jamais avoir a "finir" la fonction - indispensable pour un
            # flux infini.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        # Libere la camera proprement si la boucle s'arrete (ex. client
        # deconnecte), sinon elle resterait "occupee" par ce programme.
        camera.release()


@app.route("/video_feed")
def video_feed():
    # Cette route est appelee en continu par la Machine B (ou un navigateur).
    # mimetype="multipart/x-mixed-replace" est l'en-tete HTTP qui dit au
    # client "ce n'est pas une image fixe, chaque nouvelle partie recue
    # remplace la precedente" - c'est ce qui cree l'effet video.
    return Response(
        generate_frames(app.config["CAMERA_INDEX"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    # Page d'accueil minimale, pratique pour verifier le flux directement
    # dans un navigateur sans avoir besoin d'un autre outil.
    return "<h1>Flux webcam actif</h1><img src='/video_feed' width='720'>"


if __name__ == "__main__":
    # --- Lecture des options passees en ligne de commande ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--camera", type=int, default=0, help="Index de la webcam (0 par defaut)"
    )
    args = parser.parse_args()

    # Stocke l'index de la camera dans la config Flask pour que la route
    # /video_feed puisse le relire (elle n'a pas acces direct a "args").
    app.config["CAMERA_INDEX"] = args.camera

    local_ip = get_local_ip()
    print(f"Flux disponible sur : http://{local_ip}:{args.port}/video_feed")
    print("-> Copie cette adresse, tu en auras besoin sur la machine B.")
    print("(Verifie aussi que le pare-feu autorise ce port en entree)")

    # host="0.0.0.0" : ecoute sur TOUTES les interfaces reseau de la machine
    # (pas seulement localhost), donc accessible depuis une autre machine.
    # threaded=True : permet de gerer plusieurs clients connectes en meme
    # temps (ex. la Machine B et un navigateur de verification).
    app.run(host="0.0.0.0", port=args.port, threaded=True)
