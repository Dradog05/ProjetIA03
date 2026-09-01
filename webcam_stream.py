"""
Machine A - Diffusion du flux webcam en streaming MJPEG sur le reseau local.

Utilisation :
    python webcam_stream.py [--port 5000] [--camera 0]

Une fois lance, le flux est accessible depuis n'importe quelle machine du
meme reseau local a l'adresse : http://<IP_DE_CETTE_MACHINE>:<port>/video_feed
On peut aussi l'ouvrir directement dans un navigateur pour verifier que ca marche.
"""

import argparse
import socket

import cv2
from flask import Flask, Response

app = Flask(__name__)


def get_local_ip():
    """Recupere l'adresse IP locale de la machine sur le reseau (pour affichage)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_frames(camera_index: int):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera {camera_index}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        camera.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(app.config["CAMERA_INDEX"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return "<h1>Flux webcam actif</h1><img src='/video_feed' width='720'>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--camera", type=int, default=0, help="Index de la webcam (0 par defaut)"
    )
    args = parser.parse_args()

    app.config["CAMERA_INDEX"] = args.camera

    local_ip = get_local_ip()
    print(f"Flux disponible sur : http://{local_ip}:{args.port}/video_feed")
    print("-> Copie cette adresse, tu en auras besoin sur la machine B.")
    print("(Verifie aussi que le pare-feu autorise ce port en entree)")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
