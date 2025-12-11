# demon_game.py
import cv2
import numpy as np
import time
import random

# -----------------------------
# Config
# -----------------------------
CAMERA_INDEX = 0
DEMON_SIZE = 80               # size of corner demons (square)
MOVING_DEMON_W = 90           # moving demon width
MOVING_DEMON_H = 120          # moving demon height
CORNER_SPAWN_INTERVAL = 3.0   # seconds between corner demon spawns
MOVING_DEMON_INTERVAL = 8.0   # seconds between launches of moving demon
MOVING_DEMON_SPEED = 3        # pixels per frame
MOTION_TOUCH_THRESHOLD = 4000 # threshold for movement detection in demon area
FACE_MIN_SIZE = (60, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open camera. Is webcam available?")
    exit()

ret, frame = cap.read()
if not ret:
    print("Can't read from webcam.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Corner positions: top-left, top-right, bottom-left, bottom-right
corner_positions = {
    "tl": (10, 10),
    "tr": (width - DEMON_SIZE - 10, 10),
    "bl": (10, height - DEMON_SIZE - 10),
    "br": (width - DEMON_SIZE - 10, height - DEMON_SIZE - 10),
}

# Game state
score = 0
game_over = False
last_corner_spawn = time.time()
last_moving_spawn = time.time()
corner_demons = {}   # id -> dict {pos, born_time}
moving_demon = None   # dict {x,y,w,h,dy,active,touched_by_face}
prev_gray = None

demon_id_counter = 0

def rects_intersect(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def spawn_corner_demon(corner_key):
    global demon_id_counter
    x, y = corner_positions[corner_key]
    demon_id = demon_id_counter
    demon_id_counter += 1
    corner_demons[demon_id] = {
        "pos": (x, y),
        "born": time.time(),
        "corner": corner_key
    }

def spawn_random_corner_demon():
    corner = random.choice(list(corner_positions.keys()))
    spawn_corner_demon(corner)

def spawn_moving_demon():
    # spawn at top middle
    x = width // 2 - MOVING_DEMON_W // 2
    y = -MOVING_DEMON_H
    return {"x": x, "y": y, "w": MOVING_DEMON_W, "h": MOVING_DEMON_H, "active": True, "touched_face": False}

def detect_motion_in_region(diff_frame, rect):
    x, y, w, h = rect
    x = max(0, min(width-1, x))
    y = max(0, min(height-1, y))
    w = max(1, min(width-x, w))
    h = max(1, min(height-y, h))
    roi = diff_frame[y:y+h, x:x+w]
    # sum of white pixels after thresholding is a decent motion proxy
    motion_amount = int(np.sum(roi) / 255)  # number of changed pixels
    return motion_amount

# Initialize prev_gray
ret, frame = cap.read()
if not ret:
    print("Can't read from webcam (2). Exiting.")
    cap.release()
    exit()
frame = cv2.flip(frame, 1)  # mirror
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("Controls: 'q' to quit. When game over press 'r' to restart or 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed, exiting.")
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # motion difference
    diff = cv2.absdiff(gray, prev_gray)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Face detection on the mirrored frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=FACE_MIN_SIZE)

    # Draw face rectangle(s) and take the largest as player
    player_face = None
    if len(faces) > 0:
        # choose the largest by area
        faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        for (x, y, w, h) in faces_sorted:
            # draw a thinner rectangle for each face, highlight the main one
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
        player_face = faces_sorted[0]  # main face
        # thicker overlay to highlight player
        px, py, pw, ph = player_face
        cv2.rectangle(display, (px-3, py-3), (px+pw+3, py+ph+3), (0, 255, 255), 2)

    # Spawn corner demons periodically
    now = time.time()
    if not game_over and now - last_corner_spawn >= CORNER_SPAWN_INTERVAL:
        spawn_random_corner_demon()
        last_corner_spawn = now

    # Spawn moving demon periodically
    if moving_demon is None and not game_over and now - last_moving_spawn >= MOVING_DEMON_INTERVAL:
        moving_demon = spawn_moving_demon()
        last_moving_spawn = now

    # Draw and check corner demons
    remove_ids = []
    for did, d in list(corner_demons.items()):
        x, y = d["pos"]
        # draw as filled circle with border (a "demon")
        center = (x + DEMON_SIZE//2, y + DEMON_SIZE//2)
        cv2.rectangle(display, (x, y), (x + DEMON_SIZE, y + DEMON_SIZE), (0, 0, 255), 2)
        cv2.putText(display, "D", (x + 10, y + DEMON_SIZE - 10), FONT, 1.0, (0,0,255), 2)

        # Check for motion in demon box
        motion = detect_motion_in_region(diff_thresh, (x, y, DEMON_SIZE, DEMON_SIZE))
        # If motion amount exceeds threshold, consider it 'touched'
        if motion > MOTION_TOUCH_THRESHOLD:
            score += 1
            remove_ids.append(did)

    for rid in remove_ids:
        corner_demons.pop(rid, None)

    # Update moving demon
    if moving_demon is not None and moving_demon["active"] and not game_over:
        moving_demon["y"] += MOVING_DEMON_SPEED
        mx, my, mw, mh = moving_demon["x"], moving_demon["y"], moving_demon["w"], moving_demon["h"]
        # Draw moving demon as filled rectangle (red)
        cv2.rectangle(display, (mx, my), (mx + mw, my + mh), (0, 0, 200), -1)
        cv2.putText(display, "!" , (mx + mw//3, my + mh//2), FONT, 1.2, (255,255,255), 2)

        # If player face exists, check for collision
        if player_face is not None:
            if rects_intersect((mx, my, mw, mh), player_face):
                # Game Over
                game_over = True
                moving_demon["touched_face"] = True

        # If demon exits bottom
        if my > height:
            moving_demon["active"] = False
            # If it wasn't touched by face during trip, award 5 points
            if not moving_demon.get("touched_face", False):
                score += 5
            moving_demon = None

    # Display score
    cv2.putText(display, f"Score: {score}", (10, 30), FONT, 1.0, (255, 255, 255), 2)

    # If game over show message
    if game_over:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, height//2 - 80), (width, height//2 + 80), (0,0,0), -1)
        alpha = 0.6
        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
        cv2.putText(display, "GAME OVER", (width//2 - 200, height//2 - 10), FONT, 2.0, (0, 0, 255), 4)
        cv2.putText(display, f"Final Score: {score}", (width//2 - 180, height//2 + 40), FONT, 1.0, (255,255,255), 2)
        cv2.putText(display, "Press 'r' to restart or 'q' to quit", (width//2 - 280, height//2 + 80), FONT, 0.7, (200,200,200), 2)

    cv2.imshow("Demon Pop (mirror)", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if game_over and key == ord('r'):
        # Reset game
        score = 0
        game_over = False
        corner_demons.clear()
        moving_demon = None
        last_corner_spawn = time.time()
        last_moving_spawn = time.time()

    # save previous frame for motion detection
    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
