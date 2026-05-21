"""
DartsCamera – Flask + Flask-SocketIO server.

Endpoints:
  GET  /                      → main web interface
  GET  /video_feed            → MJPEG stream (camera)
  GET  /api/camera/frame      → single JPEG snapshot
  POST /api/camera/calibrate  → auto-detect board circle
  POST /api/camera/calibrate_manual → set board region manually
  POST /api/camera/background → set background reference frame
  GET  /api/camera/detect     → detect darts in current frame vs background
  POST /api/game/new          → start a new game
  GET  /api/game/state        → current game state
  POST /api/game/throw        → record a dart throw
  POST /api/game/end_turn     → end current player's turn
  POST /api/game/undo         → undo last dart

SocketIO events (server → client):
  game_state     – full game state dict
  dart_thrown    – result of single throw
  turn_ended     – result of turn end
  camera_status  – { available: bool }
  detection_result – list of detected darts
"""

import os
from typing import Optional

import yaml
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from core.game import Game, GameMode
from core.detector import DartDetector

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

current_game: Optional[Game] = None
detector: Optional[DartDetector] = None
config: dict = {}
_detect_active: bool = False


# ---------------------------------------------------------------------------
# Auto-detection background task
# ---------------------------------------------------------------------------

def _detection_loop() -> None:
    """Background task: detect darts every 1.5 s and emit changes via SocketIO."""
    global _detect_active
    prev_count = -1
    socketio.sleep(1.0)          # small startup delay
    while _detect_active:
        socketio.sleep(1.5)
        if not _detect_active:
            break
        if detector and detector.board_center and detector._background is not None:
            try:
                darts = detector.detect_darts()
                count = len(darts)
                if count != prev_count:
                    prev_count = count
                    socketio.emit("detection_result", {"darts": darts})
            except Exception:
                prev_count = -1


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _camera_available() -> bool:
    return detector is not None and detector.is_available()


def _mjpeg_generator():
    """Yield MJPEG frames for the /video_feed route."""
    while True:
        if detector:
            jpg = detector.get_jpeg_frame(quality=70)
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        socketio.sleep(0.05)  # ~20 fps


# ---------------------------------------------------------------------------
# Routes – pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes – video
# ---------------------------------------------------------------------------

@app.route("/video_feed")
def video_feed():
    if not _camera_available():
        return Response("Caméra non disponible", status=503)
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/camera/frame")
def camera_frame():
    if not _camera_available():
        return Response("Caméra non disponible", status=503)
    jpg = detector.get_jpeg_frame()
    if jpg is None:
        return Response("Pas d'image", status=503)
    return Response(jpg, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Routes – camera calibration
# ---------------------------------------------------------------------------

@app.route("/api/camera/calibrate", methods=["POST"])
def calibrate_auto():
    if not detector:
        return jsonify({"error": "Caméra non initialisée."}), 400
    result = detector.calibrate_auto()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/camera/calibrate_manual", methods=["POST"])
def calibrate_manual():
    if not detector:
        return jsonify({"error": "Caméra non initialisée."}), 400
    data = request.get_json(force=True) or {}
    try:
        cx = float(data["center_x"])
        cy = float(data["center_y"])
        r  = float(data["radius"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Paramètres invalides (center_x, center_y, radius)."}), 400
    detector.calibrate_manual(cx, cy, r)
    return jsonify({"status": "ok", "center": [int(cx), int(cy)], "radius": int(r)})


@app.route("/api/camera/background", methods=["POST"])
def set_background():
    if not detector:
        return jsonify({"error": "Caméra non initialisée."}), 400
    ok = detector.set_background()
    if not ok:
        return jsonify({"error": "Aucune image disponible."}), 503
    return jsonify({"status": "ok"})


@app.route("/api/camera/detect", methods=["GET"])
def detect_darts():
    if not detector:
        return jsonify({"error": "Caméra non initialisée."}), 400
    darts = detector.detect_darts()
    socketio.emit("detection_result", {"darts": darts})
    return jsonify({"darts": darts})


@app.route("/api/camera/auto_detect", methods=["POST"])
def toggle_auto_detect():
    global _detect_active
    data = request.get_json(force=True) or {}
    # If 'active' key provided use it, else toggle
    if "active" in data:
        _detect_active = bool(data["active"])
    else:
        _detect_active = not _detect_active
    if _detect_active:
        socketio.start_background_task(_detection_loop)
    socketio.emit("auto_detect_status", {"active": _detect_active})
    return jsonify({"active": _detect_active})


# ---------------------------------------------------------------------------
# Routes – game
# ---------------------------------------------------------------------------

@app.route("/api/game/new", methods=["POST"])
def new_game():
    global current_game
    data = request.get_json(force=True) or {}
    try:
        mode = GameMode(data.get("mode", "501"))
    except ValueError:
        return jsonify({"error": "Mode de jeu inconnu."}), 400

    players = [str(p).strip() for p in data.get("players", ["Joueur 1"]) if str(p).strip()]
    if not players:
        return jsonify({"error": "Aucun joueur fourni."}), 400

    double_in  = bool(data.get("double_in", False))
    double_out = bool(data.get("double_out", True))

    current_game = Game(mode=mode, player_names=players, double_in=double_in, double_out=double_out)
    state = current_game.get_state()
    socketio.emit("game_state", state)

    # Auto-calibrate once at game start (non-blocking)
    if detector and detector._current_frame is not None:
        def _do_calibrate():
            result = detector.calibrate_auto()
            socketio.emit("calibration_result", result)
        socketio.start_background_task(_do_calibrate)

    return jsonify(state)


@app.route("/api/game/state")
def game_state():
    if current_game is None:
        return jsonify({"error": "Aucune partie en cours."}), 404
    return jsonify(current_game.get_state())


@app.route("/api/game/throw", methods=["POST"])
def throw_dart():
    if current_game is None:
        return jsonify({"error": "Aucune partie en cours."}), 400
    data = request.get_json(force=True) or {}
    score  = int(data.get("score", 0))
    label  = str(data.get("label", "MISS"))
    x_norm = data.get("x_norm")
    y_norm = data.get("y_norm")

    result = current_game.throw_dart(score, label, x_norm, y_norm)
    if "error" in result:
        return jsonify(result), 400

    state = current_game.get_state()
    socketio.emit("game_state", state)
    socketio.emit("dart_thrown", result)
    return jsonify({"result": result, "state": state})


@app.route("/api/game/end_turn", methods=["POST"])
def end_turn():
    if current_game is None:
        return jsonify({"error": "Aucune partie en cours."}), 400
    result = current_game.end_turn()
    if "error" in result:
        return jsonify(result), 400
    state = current_game.get_state()
    socketio.emit("game_state", state)
    socketio.emit("turn_ended", result)
    return jsonify({"result": result, "state": state})


@app.route("/api/game/undo", methods=["POST"])
def undo_dart():
    if current_game is None:
        return jsonify({"error": "Aucune partie en cours."}), 400
    result = current_game.undo_last_dart()
    if "error" in result:
        return jsonify(result), 400
    state = current_game.get_state()
    socketio.emit("game_state", state)
    return jsonify({"result": result, "state": state})


# ---------------------------------------------------------------------------
# SocketIO events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    emit("camera_status", {"available": _camera_available()})
    if current_game:
        emit("game_state", current_game.get_state())


@socketio.on("request_detect")
def on_request_detect():
    """Client asks for dart detection."""
    if detector:
        darts = detector.detect_darts()
        emit("detection_result", {"darts": darts})
    else:
        emit("detection_result", {"darts": [], "error": "Caméra non disponible."})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()

    cam_cfg  = config.get("camera", {})
    srv_cfg  = config.get("server", {})

    cam_id   = cam_cfg.get("id", 0)
    cam_w    = cam_cfg.get("width",  1280)
    cam_h    = cam_cfg.get("height", 720)

    try:
        detector = DartDetector(camera_id=cam_id, width=cam_w, height=cam_h)
        detector.start()
        print(f"[Camera] Caméra {cam_id} initialisée ({cam_w}×{cam_h})")
    except Exception as exc:
        print(f"[Camera] Caméra non disponible : {exc}")
        print("[Camera] Lancement en mode manuel (cliquer sur la cible).")
        detector = None

    host  = srv_cfg.get("host",  "0.0.0.0")
    port  = srv_cfg.get("port",  5000)
    debug = srv_cfg.get("debug", False)

    print(f"[Server] http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
