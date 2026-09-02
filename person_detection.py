"""
Machine B - Reception du flux de la machine A, detection d'objets (YOLOv8),
et diffusion du resultat (video annotee + donnees) pour que d'autres appareils
(tablette, telephone, autre PC) puissent y acceder via un navigateur.

Prerequis : pip install -r requirements.txt
Le modele yolov8n.pt (~6 Mo) est telecharge automatiquement au premier lancement.

Utilisation :
    python person_detection.py --stream-url http://<IP_MACHINE_A>:5000/video_feed

Une fois lance, depuis n'importe quel appareil du MEME reseau local (tablette,
telephone, autre PC), ouvrir dans un navigateur :
    - Page complete (video + compteurs) : http://<IP_MACHINE_B>:5001/
    - Video annotee seule               : http://<IP_MACHINE_B>:5001/video_feed
    - Donnees brutes (JSON)             : http://<IP_MACHINE_B>:5001/api/status

Options :
    --no-window   n'affiche pas de fenetre locale sur la machine B
                  (la video reste consultable via le navigateur)
"""

import argparse
import socket
import threading
import time

import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# Classes COCO a detecter : {id_classe: "nom affiche"}.
# Pour ajouter une autre classe, il suffit d'ajouter une ligne ici
# (voir la liste complete des 80 classes COCO dans la documentation).
CLASSES_TO_DETECT = {
    0: "Personnes",
    67: "Telephones",
}

app = Flask(__name__)

# Etat partage entre le thread de detection et les routes Flask
state_lock = threading.Lock()
state = {
    "frame_bytes": None,   # derniere image annotee, encodee en JPEG
    "counts": {name: 0 for name in CLASSES_TO_DETECT.values()},
    "fps": 0.0,
    "derniere_maj": None,
    "connecte": False,
}


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def detection_loop(stream_url: str, model_name: str, conf_threshold: float, show_window: bool):
    print("Chargement du modele YOLOv8...")
    model = YOLO(model_name)

    print(f"Connexion au flux : {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    prev_time = time.time()
    class_ids = list(CLASSES_TO_DETECT.keys())

    while True:
        if not cap.isOpened():
            with state_lock:
                state["connecte"] = False
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Flux interrompu, tentative de reconnexion...")
            with state_lock:
                state["connecte"] = False
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)
            continue

        results = model(frame, classes=class_ids, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Compte le nombre de detections par classe
        counts = {name: 0 for name in CLASSES_TO_DETECT.values()}
        for cls_tensor in results[0].boxes.cls:
            name = CLASSES_TO_DETECT.get(int(cls_tensor))
            if name:
                counts[name] += 1

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        overlay_text = " | ".join(f"{name}: {n}" for name, n in counts.items())
        overlay_text += f" | FPS: {fps:.1f}"
        cv2.putText(
            annotated_frame, overlay_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state_lock:
                state["frame_bytes"] = buffer.tobytes()
                state["counts"] = counts
                state["fps"] = round(fps, 1)
                state["derniere_maj"] = time.time()
                state["connecte"] = True

        if show_window:
            cv2.imshow("Detection - Machine B", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def generate_mjpeg():
    while True:
        with state_lock:
            frame_bytes = state["frame_bytes"]
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)  # limite la diffusion a environ 30 fps


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(
            {
                "counts": state["counts"],
                "fps": state["fps"],
                "derniere_maj": state["derniere_maj"],
                "connecte": state["connecte"],
            }
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <title>Detection en direct</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        :root {
          --bg: #0f1115;
          --card: #1a1d24;
          --border: #2a2e37;
          --text: #eaeaea;
          --muted: #8a8f98;
          --accent: #4ade80;
          --accent-off: #f87171;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: var(--bg);
          color: var(--text);
          padding: 20px;
        }
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          max-width: 900px;
          margin: 0 auto 16px auto;
        }
        h1 { font-size: 1.3em; margin: 0; }
        .status {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 0.85em;
          color: var(--muted);
        }
        .dot {
          width: 9px; height: 9px; border-radius: 50%;
          background: var(--accent-off);
          transition: background 0.3s;
        }
        .dot.on { background: var(--accent); }
        .stats {
          max-width: 900px;
          margin: 0 auto 16px auto;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 10px;
        }
        .card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 14px;
          text-align: center;
        }
        .card .value { font-size: 1.8em; font-weight: 600; }
        .card .label { font-size: 0.8em; color: var(--muted); margin-top: 2px; }
        .video-wrap {
          max-width: 900px;
          margin: 0 auto;
        }
        .video-wrap img {
          width: 100%;
          display: block;
          border-radius: 10px;
          border: 1px solid var(--border);
        }
        .footer {
          max-width: 900px;
          margin: 10px auto 0 auto;
          font-size: 0.75em;
          color: var(--muted);
          text-align: right;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>Detection en direct</h1>
        <div class="status">
          <span class="dot" id="dot"></span>
          <span id="statusText">Connexion...</span>
        </div>
      </div>

      <div class="stats" id="stats"></div>

      <div class="video-wrap">
        <img src="/video_feed" alt="Flux video annote">
      </div>

      <div class="footer" id="footer"></div>

      <script>
        const icons = { "Personnes": "\\u{1F9CD}", "Telephones": "\\u{1F4F1}" };

        async function refresh() {
          try {
            const r = await fetch('/api/status');
            const data = await r.json();

            const dot = document.getElementById('dot');
            const statusText = document.getElementById('statusText');
            if (data.connecte) {
              dot.classList.add('on');
              statusText.textContent = 'En direct';
            } else {
              dot.classList.remove('on');
              statusText.textContent = 'Flux interrompu';
            }

            const statsDiv = document.getElementById('stats');
            statsDiv.innerHTML = '';
            for (const [name, count] of Object.entries(data.counts)) {
              const icon = icons[name] || '\\u{1F539}';
              statsDiv.innerHTML += `
                <div class="card">
                  <div class="value">${icon} ${count}</div>
                  <div class="label">${name}</div>
                </div>`;
            }
            statsDiv.innerHTML += `
              <div class="card">
                <div class="value">${data.fps}</div>
                <div class="label">FPS</div>
              </div>`;

            document.getElementById('footer').textContent =
              'Derniere mise a jour : ' + new Date(data.derniere_maj * 1000).toLocaleTimeString();
          } catch (e) {
            document.getElementById('dot').classList.remove('on');
            document.getElementById('statusText').textContent = 'Machine B injoignable';
          }
        }
        setInterval(refresh, 1000);
        refresh();
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", required=True, help="URL du flux MJPEG de la machine A")
    parser.add_argument("--model", default="yolov8n.pt", help="Modele YOLOv8 a utiliser")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance")
    parser.add_argument("--port", type=int, default=5001, help="Port du serveur web de la machine B")
    parser.add_argument(
        "--no-window", action="store_true",
        help="Ne pas afficher de fenetre locale sur la machine B",
    )
    args = parser.parse_args()

    t = threading.Thread(
        target=detection_loop,
        args=(args.stream_url, args.model, args.conf, not args.no_window),
        daemon=True,
    )
    t.start()

    local_ip = get_local_ip()
    print(f"Page complete (video + compteurs) : http://{local_ip}:{args.port}/")
    print(f"Video annotee seule                : http://{local_ip}:{args.port}/video_feed")
    print(f"Donnees JSON                       : http://{local_ip}:{args.port}/api/status")
    print("-> Accessible depuis une tablette, un telephone ou un autre PC du meme reseau.")
    print(f"-> Classes detectees : {', '.join(CLASSES_TO_DETECT.values())}")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
