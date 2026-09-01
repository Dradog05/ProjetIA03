Commande cmd à utiliser pour l'envoie de la vidéo : 
pip install -r requirements.txt
python webcam_stream.py


lien pour accéder à la caméra initiale en stream : 
http://10.174.254.167:5000/video_feed




Etapes à réaliser pour l'analyse IA et la transmission en Stream au client : 
Ouvre un terminal dans le dossier `machine_B_detection`, puis lance `pip install -r requirements.txt`.
Lance `python person_detection.py --stream-url http://<IP_machine_A>:5000/video_feed` en remplaçant l'IP par celle notée à l'étape 3
