# 🎯 DartsCamera

Système de caméra pour cible de fléchettes avec scoring automatique, visualisation en temps réel et gestion de parties complètes depuis une interface web.

---

## Fonctionnalités

| Fonctionnalité | Détail |
|---|---|
| **Modes de jeu** | 501, 301, Cricket, Around the Clock, Libre |
| **Détection caméra** | Soustraction de fond (OpenCV) – vue plongeante |
| **Saisie manuelle** | Clic sur la cible dessinée ou boutons Simple/Double/Triple |
| **Scoring temps réel** | WebSocket (Socket.IO) – pas de rechargement de page |
| **Suggestions de sortie** | Chemin optimal (1–3 fléchettes) pour finir en 501/301 |
| **Statistiques** | Moyenne par fléchette et par round par joueur |
| **Multi-joueurs** | Nombre illimité de joueurs, avec équipes optionnelles |
| **Cricket** | Fermeture des numéros 15-20 + Bull, tableau de bord dédié |

---

## Architecture

```
darts-camera/
├── app.py                  # Serveur Flask + Flask-SocketIO
├── config.yaml             # Configuration caméra et serveur
├── requirements.txt
├── install.sh / install.ps1
├── core/
│   ├── board.py            # Géométrie de la cible, conversion pixel → score
│   ├── checkout.py         # Calcul automatique des sorties optimales
│   ├── game.py             # Logique de jeu (tous les modes)
│   └── detector.py         # Capture caméra + détection fléchettes (OpenCV)
├── templates/
│   └── index.html          # Interface principale
└── static/
    ├── css/style.css        # Thème sombre moderne
    └── js/app.js            # Client : canvas, Socket.IO, saisie
```

---

## Installation

### Windows
```powershell
git clone https://github.com/R3coNYT/darts-camera
cd darts-camera
.\install.ps1
```

### Linux / Raspberry Pi
```bash
git clone https://github.com/R3coNYT/darts-camera
cd darts-camera
chmod +x install.sh
./install.sh
```

---

## Lancement

```bash
# Windows
.\.venv\Scripts\Activate.ps1
python app.py

# Linux
source .venv/bin/activate
python app.py
```

Ouvrez **http://localhost:5000** dans votre navigateur.

---

## Configuration caméra (`config.yaml`)

```yaml
camera:
  id: 0          # Index de la caméra (0 = webcam par défaut)
  width: 1280
  height: 720
  diff_threshold: 28   # Sensibilité de détection (0-255)
  min_contour_area: 40 # Taille minimale d'un contour (px²)
```

### Placement caméra recommandé

- Montez la caméra **en hauteur**, **dans l'axe vertical** de la cible
- Distance : 40–80 cm au-dessus de la cible
- Évitez les reflets directs sur la cible

### Calibration

1. Allez dans l'onglet **Caméra** → cliquez **Calibrer (auto)**
2. Si la détection automatique échoue, utilisez la calibration manuelle :
   - Cliquez sur **Paramètres caméra** (icône 📷)
   - Suivez les instructions pour définir le centre et le rayon de la cible
3. Cliquez **Définir fond** (plateau vide, sans fléchettes)
4. Lancez une fléchette → cliquez **Détecter fléchettes**

---

## Modes de jeu

### 501 / 301
- Les joueurs partent de 501 ou 301 points.
- Chaque fléchette soustrait son score.
- **Double-out** (activé par défaut) : la dernière fléchette doit toucher un double.
- Une suggestion de sortie optimale s'affiche en temps réel.
- **Bust** : dépasser zéro ou rester sur 1 annule le tour.

### Cricket
- Numéros actifs : 15, 16, 17, 18, 19, 20, Bull.
- Fermez un numéro en le touchant 3 fois (simple=1, double=2, triple=3 fois).
- Après fermeture, continuez à scorer tant que l'adversaire n'a pas fermé le même numéro.
- Victoire : tous les numéros fermés ET score ≥ tous les adversaires.

### Around the Clock
- Touchez les numéros de 1 à 20 dans l'ordre, puis le Bull.
- Premier à tout terminer gagne.

### Libre
- Aucune règle – enregistrement libre des scores pour s'entraîner.

---

## Suggestions de sortie

Pour chaque score ≤ 170 avec une sortie possible, l'application affiche le chemin optimal :

```
170 → T20 → T20 → DB
167 → T20 → T19 → DB
40  → D20
32  → D16
```

La dernière fléchette (le double de sortie) est mise en évidence en orange sur la cible.

---

## API REST

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/game/new` | Nouvelle partie |
| `GET`  | `/api/game/state` | État courant |
| `POST` | `/api/game/throw` | Enregistrer un lancer |
| `POST` | `/api/game/end_turn` | Valider le tour |
| `POST` | `/api/game/undo` | Annuler la dernière fléchette |
| `POST` | `/api/camera/calibrate` | Calibration automatique |
| `POST` | `/api/camera/background` | Définir le fond de référence |
| `GET`  | `/api/camera/detect` | Déclencher la détection |
| `GET`  | `/video_feed` | Flux MJPEG de la caméra |

---

## Dépendances

- Python 3.10+
- Flask 3.x
- Flask-SocketIO 5.x
- OpenCV 4.x (`opencv-python`)
- NumPy

---

## Licence

MIT – Flavien Marchand / R3coNYT