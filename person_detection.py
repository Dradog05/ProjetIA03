"""
Machine B - Reception de plusieurs flux camera (machines A), detection
d'objets (YOLOv8) sur chacun, et diffusion du resultat pour que d'autres
appareils (tablette, telephone, autre PC) puissent y acceder via un
navigateur, avec choix des classes a detecter en direct.

Prerequis : pip install -r requirements.txt

Utilisation (une ou plusieurs cameras, une option --camera par camera) :
    python person_detection.py \
        --camera "Entree=http://<IP_MACHINE_A1>:5000/video_feed" \
        --camera "Salle=http://<IP_MACHINE_A2>:5000/video_feed"

Depuis n'importe quel appareil du MEME reseau local, ouvrir dans un navigateur :
    http://<IP_MACHINE_B>:5001/

Remarque : donne des noms de camera simples (sans espace ni caractere
special), ils sont utilises directement dans les URLs.
"""

import argparse
import socket
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

# Classes COCO proposees dans l'interface : {id_classe: "nom affiche"}.
# Pour en ajouter une autre, il suffit d'ajouter une ligne ici (voir la
# liste complete des 80 classes COCO dans la documentation du projet).
AVAILABLE_CLASSES = {
    0: "Personnes",
    67: "Telephones",
    39: "Bouteilles",
}

app = Flask(__name__)

# Classes actuellement selectionnees (modifiable en direct depuis l'interface)
class_lock = threading.Lock()
active_classes = {0, 67, 39}  # Personnes + Telephones + Bouteilles par defaut

# Etat partage par camera : {nom_camera: {frame_bytes, counts, fps, ...}}
state_lock = threading.Lock()
state = {}


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


def detection_loop(cam_id: str, stream_url: str, model_name: str, conf_threshold: float):
    # Un modele charge par camera/thread : plus simple et plus sur que de
    # partager une seule instance entre plusieurs threads en parallele.
    print(f"[{cam_id}] Chargement du modele YOLOv8...")
    model = YOLO(model_name)

    print(f"[{cam_id}] Connexion au flux : {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    prev_time = time.time()

    while True:
        if not cap.isOpened():
            with state_lock:
                state[cam_id]["connecte"] = False
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_id}] Flux interrompu, tentative de reconnexion...")
            with state_lock:
                state[cam_id]["connecte"] = False
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)
            continue

        with class_lock:
            class_ids = list(active_classes)

        if class_ids:
            results = model(frame, classes=class_ids, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            counts = {}
            for cls_tensor in results[0].boxes.cls:
                cid = int(cls_tensor)
                name = AVAILABLE_CLASSES.get(cid, str(cid))
                counts[name] = counts.get(name, 0) + 1
            # Affiche aussi les classes actives a 0 si rien detecte
            for cid in class_ids:
                name = AVAILABLE_CLASSES.get(cid, str(cid))
                counts.setdefault(name, 0)
        else:
            annotated_frame = frame.copy()
            counts = {}

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        overlay = f"[{cam_id}] "
        overlay += " | ".join(f"{n}: {c}" for n, c in counts.items()) if counts else "Detection desactivee"
        overlay += f" | FPS: {fps:.1f}"
        cv2.putText(annotated_frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with state_lock:
                state[cam_id]["frame_bytes"] = buffer.tobytes()
                state[cam_id]["counts"] = counts
                state[cam_id]["fps"] = round(fps, 1)
                state[cam_id]["derniere_maj"] = time.time()
                state[cam_id]["connecte"] = True


def generate_mjpeg(cam_id: str):
    while True:
        with state_lock:
            frame_bytes = state[cam_id]["frame_bytes"]
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in state:
        return "Camera inconnue", 404
    return Response(generate_mjpeg(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        cameras_snapshot = {
            cid: {
                "counts": s["counts"],
                "fps": s["fps"],
                "derniere_maj": s["derniere_maj"],
                "connecte": s["connecte"],
            }
            for cid, s in state.items()
        }
    with class_lock:
        actives = sorted(active_classes)
    return jsonify(
        {
            "cameras": cameras_snapshot,
            "classes_disponibles": AVAILABLE_CLASSES,
            "classes_actives": actives,
        }
    )


@app.route("/api/classes", methods=["POST"])
def api_classes():
    data = request.get_json(silent=True) or {}
    requested = data.get("classes", [])
    valid_ids = set(AVAILABLE_CLASSES.keys())
    selected = {int(c) for c in requested if int(c) in valid_ids}
    with class_lock:
        active_classes.clear()
        active_classes.update(selected)
    return jsonify({"ok": True, "classes_actives": sorted(active_classes)})


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
          --bg: #0f1115; --card: #1a1d24; --border: #2a2e37;
          --text: #eaeaea; --muted: #8a8f98;
          --accent: #4ade80; --accent-off: #f87171;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0; padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: var(--bg); color: var(--text);
        }
        h1 { font-size: 1.3em; max-width: 1100px; margin: 0 auto 14px auto; }
        .settings {
          max-width: 1100px; margin: 0 auto 18px auto;
          background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 14px;
        }
        .settings h2 { font-size: 0.9em; color: var(--muted); margin: 0 0 10px 0; text-transform: uppercase; }
        .settings label {
          display: inline-flex; align-items: center; gap: 6px;
          margin: 0 16px 8px 0; font-size: 0.95em; cursor: pointer;
        }
        .cameras {
          max-width: 1100px; margin: 0 auto;
          display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
          gap: 16px;
        }
        .camera-panel {
          background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 12px;
        }
        .cam-title {
          display: flex; align-items: center; gap: 8px;
          font-weight: 600; margin-bottom: 8px;
        }
        .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-off); }
        .dot.on { background: var(--accent); }
        .camera-panel img { width: 100%; border-radius: 8px; display: block; }
        .cam-stats { margin-top: 8px; font-size: 0.85em; color: var(--muted); }
      </style>
    </head>
    <body>
      <h1>Detection en direct - multi-cameras</h1>

      <div class="settings">
        <h2>Classes a detecter</h2>
        <div id="settingsPanel">Chargement...</div>
      </div>

      <div class="cameras" id="cameras"></div>

      <script>
        let settingsBuilt = false;
        let camerasBuilt = false;

        async function refresh() {
          try {
            const r = await fetch('/api/status');
            const data = await r.json();

            if (!settingsBuilt) {
              const panel = document.getElementById('settingsPanel');
              panel.innerHTML = '';
              for (const [id, name] of Object.entries(data.classes_disponibles)) {
                const checked = data.classes_actives.includes(parseInt(id)) ? 'checked' : '';
                panel.innerHTML += `
                  <label>
                    <input type="checkbox" value="${id}" ${checked} onchange="updateClasses()">
                    ${name}
                  </label>`;
              }
              settingsBuilt = true;
            }

            if (!camerasBuilt) {
              const grid = document.getElementById('cameras');
              grid.innerHTML = '';
              for (const camId of Object.keys(data.cameras)) {
                grid.innerHTML += `
                  <div class="camera-panel">
                    <div class="cam-title">${camId} <span class="dot" id="dot-${camId}"></span></div>
                    <img src="/video_feed/${camId}">
                    <div class="cam-stats" id="stats-${camId}"></div>
                  </div>`;
              }
              camerasBuilt = true;
            }

            for (const [camId, camData] of Object.entries(data.cameras)) {
              const dot = document.getElementById('dot-' + camId);
              if (dot) dot.classList.toggle('on', camData.connecte);

              const statsDiv = document.getElementById('stats-' + camId);
              if (statsDiv) {
                const parts = Object.entries(camData.counts).map(([n, c]) => `${n}: ${c}`);
                parts.push(`FPS: ${camData.fps}`);
                statsDiv.textContent = parts.join(' | ');
              }
            }
          } catch (e) {
            console.error(e);
          }
        }

        async function updateClasses() {
          const checked = Array.from(document.querySelectorAll('#settingsPanel input:checked'))
                                .map(el => parseInt(el.value));
          await fetch('/api/classes', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({classes: checked}),
          });
        }

        setInterval(refresh, 1000);
        refresh();
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", action="append", required=True, metavar="NOM=URL",
        help="Camera a suivre, au format NOM=URL. Repeter l'option pour plusieurs cameras.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Modele YOLOv8 a utiliser")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance")
    parser.add_argument("--port", type=int, default=5001, help="Port du serveur web de la machine B")
    args = parser.parse_args()

    cameras = {}
    for entry in args.camera:
        if "=" not in entry:
            parser.error(f"Format invalide pour --camera : {entry!r} (attendu NOM=URL)")
        name, url = entry.split("=", 1)
        cameras[name] = url

    for cam_id in cameras:
        state[cam_id] = {
            "frame_bytes": None, "counts": {}, "fps": 0.0,
            "derniere_maj": None, "connecte": False,
        }

    for cam_id, url in cameras.items():
        t = threading.Thread(
            target=detection_loop, args=(cam_id, url, args.model, args.conf), daemon=True,
        )
        t.start()

    local_ip = get_local_ip()
    print(f"Tableau de bord : http://{local_ip}:{args.port}/")
    print(f"Cameras suivies : {', '.join(cameras.keys())}")
    print("-> Accessible depuis une tablette, un telephone ou un autre PC du meme reseau.")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
