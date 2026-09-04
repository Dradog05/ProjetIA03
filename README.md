# Détection d'objets multi-caméras sur réseau local

## 1. Vue d'ensemble

Le projet capture la vidéo d'une ou plusieurs webcams sur des machines
séparées, applique une détection d'objets par IA (YOLOv8) sur une machine
centrale, et rend le résultat consultable en direct depuis n'importe quel
appareil du réseau (tablette, téléphone, autre PC) via un navigateur —
sans rien installer côté client.

Les classes détectées (personnes, téléphones, bouteilles...) sont
**modifiables en direct** depuis l'interface, sans redémarrer le programme.

## 2. Architecture

```
 [Machine A1]                    [Machine B]                     [Clients]
  Webcam                          YOLOv8 + serveur web             Navigateur
     |                                 |                                 |
     |--(HTTP MJPEG, port 5000)------->|                                 |
     |   webcam_stream.py              |                                 |
                                        |--(HTTP MJPEG/JSON, port 5001)-->|
 [Machine A2]                          |    person_detection.py          |
  Webcam                                |                                 |
     |--(HTTP MJPEG, port 5000)------->|                                 |
     |   webcam_stream.py               (autant de cameras que d'options
                                          --camera passees au demarrage)
```

**Rôle de chaque composant :**

| Machine | Rôle |
|---|---|
| **Machine A** (une par webcam) | Capture sa webcam, diffuse le flux brut en HTTP. Aucune IA ici — elle ne fait que filmer et diffuser. |
| **Machine B** (une seule) | Se connecte à toutes les Machines A, fait tourner YOLOv8 sur chaque flux, republie le résultat (vidéo annotée + statistiques) via son propre serveur web. |
| **Clients** (autant que voulu) | N'importe quel appareil du réseau ouvrant un navigateur sur l'adresse de la Machine B. Ils ne font aucun calcul, juste de l'affichage. |

Toutes les machines (webcams, détection, clients) doivent être sur le
**même réseau local** — dans notre cas, un point d'accès Wi-Fi partagé
depuis un téléphone (hotspot).

## 3. Comment les composants interagissent (flux de données)

1. Chaque **Machine A** capture une image de sa webcam, l'encode en JPEG,
   et l'ajoute à un flux HTTP continu au format **MJPEG**
   (`/video_feed`, port 5000). Un flux MJPEG est simplement une succession
   d'images JPEG envoyées les unes après les autres — n'importe quel client
   HTTP (navigateur, OpenCV) peut le lire comme une vidéo.

2. La **Machine B** ouvre une connexion vers chaque flux `/video_feed` des
   Machines A (une par caméra), dans un **thread séparé par caméra**. Pour
   chaque image reçue :
   - elle la passe dans le modèle **YOLOv8**, filtré sur les classes
     actuellement sélectionnées ;
   - elle dessine les boîtes de détection et un texte récapitulatif sur
     l'image ;
   - elle stocke le résultat (image annotée + compteurs) dans une variable
     partagée, protégée par un verrou (`threading.Lock`) pour éviter que le
     thread de détection et le serveur web ne se marchent dessus en la
     lisant/écrivant en même temps.

3. Le **serveur web de la Machine B** (Flask, port 5001) expose ce résultat
   via trois types de routes :
   - `/video_feed/<nom_camera>` — le flux vidéo annoté d'une caméra donnée ;
   - `/api/status` — les statistiques de toutes les caméras et la liste des
     classes actives, en JSON ;
   - `/api/classes` (en écriture, POST) — permet de changer les classes à
     détecter à chaud ;
   - `/api/fall-detection` (en écriture, POST) — active/désactive la
     détection de chute à chaud (voir section 7).

4. N'importe quel **client** (tablette, téléphone, autre PC) ouvre
   `http://<IP_MACHINE_B>:5001/` dans un navigateur. La page interroge
   `/api/status` toutes les secondes (JavaScript) pour rafraîchir les
   compteurs, et affiche chaque flux vidéo directement via son URL
   `/video_feed/<nom_camera>`. Quand une case à cocher (classe à détecter)
   est modifiée, le navigateur envoie la nouvelle sélection à
   `/api/classes` ; la Machine B en tient compte dès l'image suivante.

Point important : **aucun calcul ne se fait côté client**. Un téléphone ou
une tablette n'a besoin que d'un navigateur — toute l'IA tourne sur la
Machine B.

## 4. Structure du projet

```
machine_A_streaming/
  webcam_stream.py     -> capture + diffusion d'une webcam (port 5000)
  requirements.txt     -> flask, opencv-python
  (a copier et lancer sur CHAQUE machine webcam)

machine_B_detection/
  person_detection.py  -> reception des flux, detection YOLOv8, serveur web (port 5001)
  requirements.txt     -> opencv-python, ultralytics, flask
  (une seule machine)

README.md                    -> ce document
documentation_latex/         -> rapport de projet complet (.tex + .pdf)
```

## 5. Installation

Sur **chaque machine webcam** (Machine A) :
```bash
cd machine_A_streaming
pip install -r requirements.txt
```

Sur la **machine de détection** (Machine B) :
```bash
cd machine_B_detection
pip install -r requirements.txt
```

## 6. Guide de démarrage — qui fait quoi, sur quelle machine

Ordre à respecter : téléphone → machines webcam → machine de détection →
clients. Les adresses IP ci-dessous sont celles de notre déploiement de
test ; elles peuvent changer si le hotspot redistribue de nouvelles IP —
vérifie toujours ce qu'affiche réellement chaque terminal.

### Téléphone (hotspot + caméra)
1. Activer le partage de connexion Wi-Fi (hotspot) sur ce téléphone.
2. Ouvrir l'application de caméra IP (ex. **IP Webcam** sur Android) et
   appuyer sur **Start server**.
3. Noter l'adresse affichée — chez nous : `10.174.254.68:8080`.

Ce téléphone n'a rien d'autre à faire : il reste allumé, hotspot actif,
app ouverte. Pas besoin d'installer Python ni aucun des fichiers du projet
dessus.

### Chaque machine webcam (Machine A)
Se connecter au hotspot, puis :
```powershell
cd machine_A_streaming
python webcam_stream.py
```
Noter l'adresse affichée dans le terminal — chez nous :
`10.174.254.167:5000` (Machine A1) et `10.174.254.226:5000` (Machine A2).
Laisser le terminal ouvert.

### Machine de détection (Machine B) — une seule, à lancer en dernier
Se connecter au hotspot, puis, sur **une seule ligne** (PowerShell
n'accepte pas le `\` de continuation Bash) :
```powershell
cd machine_B_detection
python person_detection.py --camera "Entree=http://10.174.254.167:5000/video_feed" --camera "Salle=http://10.174.254.226:5000/video_feed" --camera "Telephone=http://10.174.254.68:8080/video"
```
Le terminal affiche l'adresse du tableau de bord à utiliser — chez nous :
```
Tableau de bord : http://10.174.254.171:5001/
```

Options disponibles :
| Option | Rôle | Défaut |
|---|---|---|
| `--camera NOM=URL` | Caméra à suivre (répétable, une par source) | obligatoire |
| `--model` | Fichier du modèle YOLOv8 | `yolov8n.pt` |
| `--conf` | Seuil de confiance des détections | `0.5` |
| `--port` | Port du serveur web de la Machine B | `5001` |

### Clients (tablette, téléphone, autre PC — autant que voulu)
Rien à installer, rien à lancer. Juste se connecter au même hotspot et
ouvrir un navigateur sur l'adresse donnée par la Machine B, par exemple :
```
http://10.174.254.171:5001
```
Le tableau de bord affiche les trois caméras côte à côte, avec les cases
à cocher (classes à détecter, détection de chute) et les statistiques en
direct.

## 7. Fonctionnalités

### Sélection dynamique des classes
En haut du tableau de bord, des cases à cocher permettent de choisir quelles
classes détecter, y compris plusieurs en même temps (ex. Personnes +
Bouteilles). Le changement est envoyé au serveur et pris en compte
immédiatement, sans redémarrage. Classes actuellement disponibles (toutes
activées par défaut) : Personnes, Téléphones, Bouteilles.

À noter : la classe COCO `bottle` est une bouteille générique (le modèle
pré-entraîné ne distingue pas une bouteille d'eau d'un autre type de
bouteille — cette granularité demanderait un modèle réentraîné sur des
images annotées spécifiquement).

Pour ajouter une autre classe COCO, il suffit d'ajouter une ligne dans le
dictionnaire `AVAILABLE_CLASSES` en haut de `person_detection.py` (voir la
documentation pour la liste complète des 80 classes COCO).

### Multi-caméras
Chaque caméra tourne dans son propre thread avec son propre modèle YOLOv8
chargé indépendamment (plus simple et plus robuste que de partager un seul
modèle entre plusieurs threads). Le tableau de bord affiche tous les flux
côte à côte, chacun avec son propre indicateur de connexion (point vert /
rouge) et ses propres statistiques.

### Détection de chute (expérimental)
Une case à cocher séparée ("Détection de chute") active, en plus du modèle
de détection d'objets, un second modèle spécialisé (`yolov8n-pose.pt`) qui
détecte le squelette des personnes. Une chute est détectée par une
heuristique simple (boîte englobante nettement plus large que haute),
confirmée seulement si elle persiste sur au moins 60% des 10 dernières
images (pour éviter les fausses alertes, ex. une personne qui se penche).
Désactivée par défaut, car elle double la charge de calcul (deux modèles
tournent en parallèle sur chaque caméra). Quand une chute est confirmée, un
bandeau rouge clignotant apparaît sur le panneau de la caméra concernée.

## 8. Mise en réseau et dépannage

Toutes les machines doivent être connectées au même point d'accès Wi-Fi
(hotspot téléphone dans notre cas).

| Problème | Cause | Solution |
|---|---|---|
| `ping` en échec entre deux machines alors que tout fonctionne | Windows classe les hotspots en profil réseau "Public", qui bloque par défaut les requêtes ICMP entrantes | Ignorer le ping, tester directement le flux HTTP réel ; ou repasser le réseau en "Privé" dans les paramètres Wi-Fi |
| Flux inaccessible depuis une autre machine | Pare-feu bloquant le port (5000 ou 5001) | Accepter la popup Windows d'autorisation réseau au premier lancement, ou passer le réseau en "Privé" |
| `Camera inconnue` (404) sur `/video_feed/<nom>` | Nom de caméra mal orthographié dans l'URL | Vérifier le nom exact utilisé dans `--camera NOM=URL` |
| FPS bas | Normal sur CPU avec plusieurs caméras en parallèle | Réduire le nombre de classes actives, ou la résolution de capture côté Machine A |
| Erreur `Jeton inattendu` / `MissingExpressionAfterOperator` au lancement | Commande écrite sur plusieurs lignes avec `\` (syntaxe Bash), incompatible avec PowerShell | Écrire la commande `--camera ...` entière sur une seule ligne |
| App caméra IP affiche une erreur "pas de Wi-Fi" sur le téléphone-hotspot | Certaines apps attendent que le téléphone soit connecté en tant que client Wi-Fi, pas seulement en mode point d'accès | Généralement fonctionne quand même (le téléphone garde une IP sur son propre réseau) ; sinon, utiliser un second téléphone dédié à la caméra |

