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

import argparse   # lecture des options passees en ligne de commande (--camera, --port...)
import socket     # utilise dans get_local_ip() pour trouver l'adresse IP de la machine
import threading  # un thread independant par camera + verrous pour les donnees partagees
import time       # mesure du temps (FPS) et pauses entre les tentatives de reconnexion

import torch
import cv2                                    # OpenCV : lecture du flux, encodage JPEG
from flask import Flask, Response, jsonify, request  # mini-serveur web (routes HTTP, JSON)
from ultralytics import YOLO                  # chargement et execution du modele YOLOv8

# Classes COCO proposees dans l'interface : {id_classe: "nom affiche"}.
# Pour en ajouter une autre, il suffit d'ajouter une ligne ici (voir la
# liste complete des 80 classes COCO dans la documentation du projet).
AVAILABLE_CLASSES = {
    0: "Personnes",
    67: "Telephones",
    39: "Bouteilles",
}

# --- Paramètres d'optimisation de performance ---
# Exécuter YOLO toutes les N frames (1 = chaque frame, 2 = 1/2, 3 = 1/3)
FRAME_SKIP = 2

# Taille de redimensionnement pour l'inférence YOLO (défaut Ultralytics : 640)
# 480 ou 384 offre un excellent ratio fluidité / précision
INFERENCE_SIZE = 480

# Qualité de réencodage JPEG pour le flux web (de 0 à 100, défaut OpenCV : 95)
# 60 permet d'économiser beaucoup de CPU et de bande passante Wi-Fi
JPEG_QUALITY = 60

# Cree l'application web. Flask va gerer tout le protocole HTTP a notre
# place ; il ne reste plus qu'a definir une fonction par URL ("route").
app = Flask(__name__)

# Classes actuellement selectionnees (modifiable en direct depuis l'interface)
# `class_lock` protege `active_classes` : plusieurs threads (chaque camera,
# plus le serveur web) peuvent le lire/modifier en meme temps, le verrou
# garantit qu'un seul a la fois y accede reellement.
class_lock = threading.Lock()
active_classes = {0, 67, 39}  # Personnes + Telephones + Bouteilles par defaut

# Etat partage par camera : {nom_camera: {frame_bytes, counts, fps, ...}}
# C'est ici que chaque thread de detection depose son dernier resultat,
# et que le serveur web va le relire pour repondre aux clients.
state_lock = threading.Lock()
state = {}

def get_local_ip():
    # Astuce : on "fait semblant" de se connecter a une adresse externe
    # (8.8.8.8) juste pour demander au systeme d'exploitation quelle
    # adresse IP locale serait utilisee pour cette connexion. Aucune
    # donnee n'est reellement envoyee sur le reseau.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"  # repli si la machine n'a pas de reseau du tout
    finally:
        s.close()
    return ip


def detection_loop(cam_id: str, stream_url: str, model_name: str, conf_threshold: float):
    """Boucle infinie exécutée dans son propre thread pour une caméra.

    Optimisations intégrées :
    - Sélection du device PyTorch le plus rapide (CUDA > MPS > CPU)
    - Inférence YOLO espacée de FRAME_SKIP frames avec conservation de l'image annotée native YOLO
    - Buffer OpenCV à 1 image pour supprimer le lag
    - Réencodage JPEG allégé
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[{cam_id}] Modèle YOLOv8 chargé sur : {device.upper()}")
    model = YOLO(model_name)
    model.to(device)

    print(f"[{cam_id}] Connexion au flux : {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = time.time()
    frame_counter = 0

    # Mémorisation de la dernière image annotée avec le style natif YOLO
    last_annotated_frame = None
    counts = {}

    while True:
        # --- Gestion de la reconnexion si le flux tombe ---
        if not cap.isOpened():
            with state_lock:
                state[cam_id]["connecte"] = False
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_id}] Flux interrompu, tentative de reconnexion...")
            with state_lock:
                state[cam_id]["connecte"] = False
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            continue

        frame_counter += 1

        with class_lock:
            class_ids = list(active_classes)

        if class_ids:
            # Inférence exécutée toutes les FRAME_SKIP frames
            if frame_counter % FRAME_SKIP == 0 or last_annotated_frame is None:
                results = model.predict(
                    source=frame,
                    classes=class_ids,
                    conf=conf_threshold,
                    imgsz=INFERENCE_SIZE,
                    device=device,
                    verbose=False,
                )

                # Mise à jour des compteurs
                counts = {}
                for cls_tensor in results[0].boxes.cls:
                    cid = int(cls_tensor)
                    name = AVAILABLE_CLASSES.get(cid, str(cid))
                    counts[name] = counts.get(name, 0) + 1
                for cid in class_ids:
                    name = AVAILABLE_CLASSES.get(cid, str(cid))
                    counts.setdefault(name, 0)

                # .plot() génère automatiquement les couleurs COCO d'origine et la précision
                last_annotated_frame = results[0].plot()

            # Utilise le rendu natif YOLO (frais ou mémorisé)
            annotated_frame = last_annotated_frame.copy()
        else:
            # Aucune classe cochée : image brute
            annotated_frame = frame.copy()
            last_annotated_frame = None
            counts = {}

        # --- Mesure du FPS réel ---
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        # --- Bandeau récapitulatif supérieur ---
        overlay = f"[{cam_id}] "
        overlay += " | ".join(f"{n}: {c}" for n, c in counts.items()) if counts else "Detection desactivee"
        overlay += f" | FPS: {fps:.1f}"
        cv2.putText(annotated_frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Encodage JPEG et mise à jour de l'état partagé ---
        ok, buffer = cv2.imencode(
            ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if ok:
            with state_lock:
                state[cam_id]["frame_bytes"] = buffer.tobytes()
                state[cam_id]["counts"] = counts
                state[cam_id]["fps"] = round(fps, 1)
                state[cam_id]["derniere_maj"] = time.time()
                state[cam_id]["connecte"] = True

def generate_mjpeg(cam_id: str):
    # Generateur MJPEG lu par un client (navigateur/tablette). Contrairement
    # a la machine A qui lit une vraie camera, ici on relit simplement la
    # DERNIERE image deja calculee par detection_loop() dans `state` :
    # plusieurs clients peuvent donc regarder la meme camera sans faire
    # tourner la detection plusieurs fois.
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
        time.sleep(0.03)  # limite la diffusion a environ 30 fps


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    # Flux video annote d'UNE camera donnee (identifiee par son nom).
    if cam_id not in state:
        return "Camera inconnue", 404
    return Response(generate_mjpeg(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    # Etat courant de toutes les cameras + classes actives, en JSON.
    # C'est cette route que le JavaScript du tableau de bord interroge
    # toutes les secondes pour rafraichir les compteurs affiches.
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
    # Change en direct les classes d'objets a detecter (appelee par le
    # JavaScript quand l'utilisateur coche/decoche une case). Prend effet
    # des l'image suivante dans chaque detection_loop().
    data = request.get_json(silent=True) or {}
    requested = data.get("classes", [])
    valid_ids = set(AVAILABLE_CLASSES.keys())
    # Ignore silencieusement tout identifiant qui ne serait pas une classe
    # proposee (protection basique contre une requete malformee).
    selected = {int(c) for c in requested if int(c) in valid_ids}
    with class_lock:
        active_classes.clear()
        active_classes.update(selected)
    return jsonify({"ok": True, "classes_actives": sorted(active_classes)})


@app.route("/")
def index():
    # Tableau de bord HTML complet : reglages (classes) + panneaux video + JS.
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <title>Detection en direct</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        /* Variables de couleur (theme sombre) */
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
        /* Bloc de reglages (cases a cocher des classes) */
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
        /* Grille des panneaux camera : s'adapte automatiquement a la largeur d'ecran */
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
        .dot.on { background: var(--accent); }  /* vert quand la camera est connectee */
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
        // Ces "drapeaux" evitent de reconstruire les cases a cocher et les
        // panneaux camera a CHAQUE rafraichissement : une fois construits,
        // on se contente de mettre a jour leur contenu (sinon un clic de
        // l'utilisateur pourrait etre efface par le rafraichissement suivant).
        let settingsBuilt = false;
        let camerasBuilt = false;

        // Appelee toutes les secondes : va chercher l'etat courant sur le
        // serveur et met a jour l'affichage.
        async function refresh() {
          try {
            const r = await fetch('/api/status');
            const data = await r.json();

            // Construit une fois les cases a cocher des classes disponibles
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

            // Construit une fois un panneau par camera (image + stats)
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

            // Met a jour, pour chaque camera : le point de connexion et le
            // texte des statistiques.
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

        // Envoie au serveur la liste des classes actuellement cochees.
        async function updateClasses() {
          const checked = Array.from(document.querySelectorAll('#settingsPanel input:checked'))
                                .map(el => parseInt(el.value));
          await fetch('/api/classes', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({classes: checked}),
          });
        }

        setInterval(refresh, 1000);  // rafraichissement automatique chaque seconde
        refresh();                   // premier appel immediat au chargement
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    # --- Lecture des options passees en ligne de commande ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", action="append", required=True, metavar="NOM=URL",
        help="Camera a suivre, au format NOM=URL. Repeter l'option pour plusieurs cameras.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Modele YOLOv8 a utiliser")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance")
    parser.add_argument("--port", type=int, default=5001, help="Port du serveur web de la machine B")
    args = parser.parse_args()

    # Transforme la liste d'options --camera "NOM=URL" en dictionnaire
    # {nom: url}, une entree par camera declaree en ligne de commande.
    cameras = {}
    for entry in args.camera:
        if "=" not in entry:
            parser.error(f"Format invalide pour --camera : {entry!r} (attendu NOM=URL)")
        name, url = entry.split("=", 1)
        cameras[name] = url

    # Initialise l'etat partage AVANT de demarrer les threads, pour que
    # les routes Flask (ex. video_feed) trouvent toujours une entree valide
    # meme si elles sont appelees avant la premiere image traitee.
    for cam_id in cameras:
        state[cam_id] = {
            "frame_bytes": None, "counts": {}, "fps": 0.0,
            "derniere_maj": None, "connecte": False,
        }

    # Un thread de detection independant par camera, demarre en arriere-plan
    # (daemon=True : ces threads s'arretent automatiquement si le programme
    # principal se termine, pas besoin de les stopper manuellement).
    for cam_id, url in cameras.items():
        t = threading.Thread(
            target=detection_loop, args=(cam_id, url, args.model, args.conf), daemon=True,
        )
        t.start()

    local_ip = get_local_ip()
    print(f"Tableau de bord : http://{local_ip}:{args.port}/")
    print(f"Cameras suivies : {', '.join(cameras.keys())}")
    print("-> Accessible depuis une tablette, un telephone ou un autre PC du meme reseau.")

    # Demarre le serveur web (thread principal). host="0.0.0.0" : accessible
    # depuis les autres machines du reseau, pas seulement en local.
    app.run(host="0.0.0.0", port=args.port, threaded=True)
