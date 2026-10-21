"""
crazyflie.py

WASD-controlled Crazyflie (with Q/E or Left/Right-arrow yaw to turn and
scan the room) whose YOLO-World camera feed detects industrial machines
(robotic arms, conveyor belts, ...) in the scene. A second Tk window shows
the identified machine's real-world model and spec sheet.
"""

from controller import Supervisor, Keyboard
from math import cos, sin, atan2, pi
import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import tkinter as tk

from pid_controller import pid_velocity_fixed_height_controller
from spare_parts_agent_bridge import AgentBridge
from agent_hud import draw_agent_hud

FLYING_ATTITUDE = 1.0

# Open-vocabulary classes to look for -- stock YOLO/COCO has no "machine"
# category, so YOLO-World is prompted with the specific things in this world.
# NOTE: "fruit sorting machine" was tried here first and had to be dropped --
# the word "fruit" in the prompt makes YOLO-World's text embedding match
# orange/apple-colored clusters directly, so it was firing on the fruit
# itself, not the arm. Keep machine prompts fruit-word-free.
MACHINE_CLASSES = [
    "robotic arm", "conveyor belt", "industrial machine", "factory machine",
    "robot", "SCARA robot", "yellow robot arm", "manufacturing robot",
    "assembly robot", "mechanical arm",
]

# Distractor classes: never shown/counted (see the machine_set filter in
# yolo_worker below), they exist purely to give YOLO-World's classifier
# somewhere else to put the fruit/crate clusters it's very confident about
# (~0.1-0.5 conf) so that confidence mass doesn't leak into the machine
# classes above, which score far lower (~0.003-0.02) even on a true hit.
DISTRACTOR_CLASSES = ["orange fruit", "apple fruit", "pile of fruit", "wooden crate"]

# YOLO-World's zero-shot confidence on Webots' synthetic renders runs far
# below its defaults (real machine detections here score ~0.003-0.02, not
# >0.25), so the usual conf=0.25 default never fires a single box. These are
# tuned against this world: low enough to catch the machine, high enough
# that an empty wall/floor crop still yields zero boxes.
YOLO_CONF = 0.002
YOLO_IOU = 0.4
MAX_DETECTIONS_SHOWN = 5

# The Webots ScaraT6 proto (projects/robots/epson/scara_t6) is an explicit
# model of a real production robot -- its own header identifies it as the
# "T6-602S Epson Scara robot" -- so this is the real machine's datasheet,
# not a fabrication. Figures are the T6-602S / T6-B602S (integrated
# controller variant) specs as published by Epson and its distributors
# (epson.com, rbtx.com, unchainedrobotics.de), current as of Aug 2026.
MACHINE_INFO = {
    "identified_as": "Epson SCARA T6-602S",
    "manufacturer": "Seiko Epson Corporation",
    "family": "Epson T-Series SCARA (4-axis)",
    "role_in_scene": "Pick-and-place fruit sorting arm with vacuum end effector,"
                      " feeding sorted crates from the conveyor belt",
    "is_real_machine": True,
    "source_note": "Specs are the manufacturer's published T6-602S/T6-B602S "
                    "datasheet values; the Webots proto's own header names "
                    "this exact model.",
    "specs": [
        ("Axes / DOF", "4 (SCARA: 3 rotary + 1 linear Z)"),
        ("Max payload", "6 kg"),
        ("Reach", "600 mm"),
        ("Repeatability", "± 0.04 mm"),
        ("Robot mass", "~22 kg"),
        ("Max TCP speed", "4.18 m/s"),
        ("Std. cycle time", "from 0.46 s (pick & place)"),
        ("J1 range", "± 132°"),
        ("J2 range", "± 150°"),
        ("J3 (Z) range", "200 mm"),
        ("J4 range", "± 360°"),
        ("Controller", "Integrated all-in-one, Epson RC+ / RC+ Express"),
        ("Power", "100-240 V AC"),
        ("Typical use", "Material handling, pick & place, packaging, food industry"),
    ],
}


class MachineInfoPanel:
    """Second GUI window: identifies the machine YOLO-World is looking for
    and shows its real-world spec sheet, with a live detection status line
    driven by the same yolo_result the camera overlay uses."""

    def __init__(self, agent_bridge=None):
        self.agent_bridge = agent_bridge
        self.root = tk.Tk()
        self.root.title("Machine Identification")
        self.root.geometry("440x560")
        self.root.configure(bg="#1e1e1e")

        header = tk.Label(self.root, text=MACHINE_INFO["identified_as"],
                           font=("Segoe UI", 15, "bold"),
                           fg="#ffb400", bg="#1e1e1e", pady=8)
        header.pack(fill="x")

        subtitle = tk.Label(self.root, text=MACHINE_INFO["manufacturer"] +
                             "  |  " + MACHINE_INFO["family"],
                             font=("Segoe UI", 10), fg="#cccccc", bg="#1e1e1e")
        subtitle.pack(fill="x", pady=(0, 4))

        role = tk.Label(self.root, text=MACHINE_INFO["role_in_scene"],
                         font=("Segoe UI", 9, "italic"), fg="#8fd3ff",
                         bg="#1e1e1e", wraplength=420, justify="left")
        role.pack(fill="x", padx=10, pady=(0, 8))

        self.status_var = tk.StringVar(value="Waiting for first camera frame...")
        status = tk.Label(self.root, textvariable=self.status_var,
                           font=("Consolas", 10, "bold"), fg="#00ff88",
                           bg="#111111", anchor="w", padx=8, pady=6)
        status.pack(fill="x", padx=10, pady=(0, 10))

        # Spec sheet only shown while the arm is actually in frame (YOLO or
        # ground-truth assist) - built once, shown/hidden via pack/pack_forget
        # in _poll() rather than rebuilt each time (cheaper, and avoids
        # flicker from destroying/recreating widgets every 200ms).
        self.detail_frame = tk.Frame(self.root, bg="#1e1e1e")

        table = tk.Frame(self.detail_frame, bg="#1e1e1e")
        table.pack(fill="both", expand=True, padx=10)
        for i, (label, value) in enumerate(MACHINE_INFO["specs"]):
            tk.Label(table, text=label, font=("Segoe UI", 9, "bold"),
                     fg="#ffffff", bg="#1e1e1e", anchor="w"
                     ).grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(table, text=value, font=("Segoe UI", 9),
                     fg="#dddddd", bg="#1e1e1e", anchor="w"
                     ).grid(row=i, column=1, sticky="w", padx=(10, 0), pady=2)

        source = tk.Label(self.detail_frame, text=MACHINE_INFO["source_note"],
                           font=("Segoe UI", 8), fg="#888888", bg="#1e1e1e",
                           wraplength=420, justify="left", pady=8)
        source.pack(fill="x", padx=10, pady=(8, 0))

        self.waiting_frame = tk.Frame(self.root, bg="#1e1e1e")
        tk.Label(self.waiting_frame, text="Machine not currently in frame",
                 font=("Segoe UI", 10), fg="#666666", bg="#1e1e1e", pady=30
                 ).pack(fill="both", expand=True)
        self._detail_shown = None  # tri-state so the first _poll() always sets it once
        self.waiting_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        agent_header = tk.Label(self.root, text="SPARE-PARTS AGENT",
                                 font=("Segoe UI", 9, "bold"), fg="#a78bfa",
                                 bg="#1e1e1e", anchor="w", pady=4)
        agent_header.pack(fill="x", padx=10, pady=(6, 0))
        self.agent_status_var = tk.StringVar(value="Agent bridge not attached.")
        agent_status = tk.Label(self.root, textvariable=self.agent_status_var,
                                 font=("Consolas", 9), fg="#c4b5fd", bg="#111111",
                                 anchor="w", padx=8, pady=6, wraplength=420, justify="left")
        agent_status.pack(fill="x", padx=10, pady=(0, 10))

        self._poll()
        self._poll_agent()

    def _poll(self):
        with yolo_lock:
            dets = list(yolo_result)
            gt_active = gt_detection_active
        detected = bool(dets) or gt_active

        if dets:
            best = max(dets, key=lambda d: d[5])
            self.status_var.set(
                f"DETECTED  ->  \"{best[4]}\"  conf={best[5]:.3f}  "
                f"({len(dets)} box(es) in frame)")
        elif gt_active:
            self.status_var.set("DETECTED  ->  ground-truth assist (YOLO found nothing this frame)")
        else:
            self.status_var.set("No detection in current camera frame")

        # Only show/hide when the state actually changed - avoids
        # unnecessary pack/pack_forget churn every 200ms poll.
        if detected != self._detail_shown:
            if detected:
                self.waiting_frame.pack_forget()
                self.detail_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            else:
                self.detail_frame.pack_forget()
                self.waiting_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self._detail_shown = detected

        self.root.after(200, self._poll)

    def _poll_agent(self):
        if self.agent_bridge is not None:
            with self.agent_bridge.lock:
                st = dict(self.agent_bridge.state)
            self.agent_status_var.set(f"[{st['phase'].upper()}]  {st['detail']}")
        self.root.after(300, self._poll_agent)

    def run(self):
        self.root.mainloop()

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Ground-truth fallback: this controller already runs in Supervisor mode
# (full read access to the scene graph, not just camera pixels), so when
# YOLO-World's zero-shot detection doesn't fire on a given frame, real
# simulation geometry - the arm's actual world position vs. the drone's
# actual position/heading - can confirm it's genuinely in view instead of
# leaving the pipeline with nothing to act on. This is not a fabricated
# detection: it's Webots' own ground truth, clearly labeled as such in
# every downstream state field so it's never confused with a real YOLO box.
scara_node = robot.getFromDef("SCARA_ARM")
if scara_node is None:
    print("WARNING: DEF 'SCARA_ARM' not found in the world - ground-truth "
          "detection fallback disabled (reload the world if you just added it).")

GT_MAX_RANGE_M = 4.0
GT_FOV_COS_THRESHOLD = 0.3  # generous ~72 degree half-angle cone


def ground_truth_arm_visible(drone_x, drone_y, drone_yaw, arm_node):
    """Best-effort geometric check, not a substitute for real perception -
    used only as a fallback trigger when YOLO-World finds nothing on a given
    frame, so the agent pipeline still gets exercised. Assumes the camera
    looks along the drone's body +X axis at yaw=0, the same forward-vector
    convention already used for the body-frame velocity transform above;
    if that assumption is off, worst case this fires a little early/late,
    not incorrectly - the range+cone margins are deliberately generous.
    """
    if arm_node is None:
        return False
    arm_x, arm_y, _ = arm_node.getPosition()
    dx, dy = arm_x - drone_x, arm_y - drone_y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6 or dist > GT_MAX_RANGE_M:
        return False
    dot = (dx / dist) * cos(drone_yaw) + (dy / dist) * sin(drone_yaw)
    return dot > GT_FOV_COS_THRESHOLD

keyboard = Keyboard()
keyboard.enable(timestep)

# Optional hands-free flight for demo recordings: set CF_AUTOPILOT=1 before
# launching Webots. Keyboard control is unchanged when it's unset.
AUTOPILOT = os.environ.get("CF_AUTOPILOT") == "1"

# Optional demo capture: CF_RECORD_DIR=<folder> records the 3D view via
# Webots' movie recorder plus the HUD camera feed (sampled on the same
# simulation clock) for CF_RECORD_SECONDS, then pauses the simulation.
RECORD_DIR = os.environ.get("CF_RECORD_DIR")
RECORD_SECONDS = float(os.environ.get("CF_RECORD_SECONDS", "60"))
RECORD_FPS = 25
hud_writer = None
show_window = True
next_hud_t = 0.0
recording_done = False
if RECORD_DIR:
    os.makedirs(RECORD_DIR, exist_ok=True)
    robot.movieStartRecording(os.path.join(RECORD_DIR, "scene.mp4"), 1280, 720, 0, 95, 1, False)


def autopilot_command(t, x, y, yaw, arm_node):
    """Scripted demo flight: take off, slow 360-degree scan of the room,
    then turn to face the SCARA arm, close in, and slowly orbit it."""
    if t < 4.0:
        return 0.0, 0.0, 0.0  # climb to cruise height and settle
    if t < 17.0:
        return 0.0, 0.0, 0.5  # scan the room
    if arm_node is None:
        return 0.0, 0.0, 0.0
    ax, ay, _ = arm_node.getPosition()
    dx, dy = ax - x, ay - y
    dist = (dx * dx + dy * dy) ** 0.5
    err = (atan2(dy, dx) - yaw + pi) % (2 * pi) - pi
    yaw_cmd = max(-0.8, min(0.8, 2.0 * err))
    forward = 0.0
    if abs(err) < 0.35:
        forward = max(-0.2, min(0.3, 0.4 * (dist - 1.1)))
    sideways = 0.15 if t > 32.0 and abs(err) < 0.35 else 0.0
    return forward, sideways, yaw_cmd

# Autonomous follow (phase 3 of the mission): the same image-space loop as
# Desktop/Projects/Drone Follow/follow.py -- target offset drives yaw, target
# box height drives forward speed (distance hold). If YOLO misses the worker
# on a frame, the supervisor's ground-truth position keeps the loop closed
# (labelled GT ASSIST on the HUD, same policy as the machine detection).
worker_node = robot.getFromDef("WORKER")
FOLLOW_START_T = 35.0
FOLLOW_SIZE = 0.55          # person box height / frame height to hold
FOLLOW_STANDOFF_M = 1.4
follow_toggle = False       # manual mode: F key


def follow_command(person_box, frame_w, frame_h, x, y, yaw):
    """Returns (forward, sideways, yaw_rate, source)."""
    if person_box is not None:
        x1, y1, x2, y2 = person_box
        ex = ((x1 + x2) / 2 - frame_w / 2) / (frame_w / 2)    # + = right
        size = (y2 - y1) / frame_h
        yaw_cmd = max(-1.0, min(1.0, -1.6 * ex))
        forward = max(-0.25, min(0.4, 1.2 * (FOLLOW_SIZE - size))) if abs(ex) < 0.6 else 0.0
        return forward, 0.0, yaw_cmd, "YOLO"
    if worker_node is None:
        return 0.0, 0.0, 0.4, "SEARCH"
    wx, wy, _ = worker_node.getPosition()
    dx, dy = wx - x, wy - y
    dist = (dx * dx + dy * dy) ** 0.5
    err = (atan2(dy, dx) - yaw + pi) % (2 * pi) - pi
    yaw_cmd = max(-1.0, min(1.0, 2.0 * err))
    forward = max(-0.25, min(0.4, 0.5 * (dist - FOLLOW_STANDOFF_M))) if abs(err) < 0.5 else 0.0
    return forward, 0.0, yaw_cmd, "GT ASSIST"


# Motors
m1_motor = robot.getDevice("m1_motor")
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor")
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor")
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor")
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(1)

# Sensors
imu = robot.getDevice("inertial_unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)

# YOLO-World runs in a background thread so inference latency never stalls the sim step.
print("Loading YOLO-World model (first run downloads the weights, needs internet)...")
model = YOLO("yolov8s-worldv2.pt")
model.set_classes(MACHINE_CLASSES + DISTRACTOR_CLASSES)
machine_class_set = set(MACHINE_CLASSES)

yolo_frame = None
yolo_result = []
gt_detection_active = False  # this-frame ground-truth-assist state, for the Tk panel/HUD
yolo_lock = threading.Lock()


def yolo_worker():
    global yolo_frame, yolo_result
    while True:
        with yolo_lock:
            frame = yolo_frame.copy() if yolo_frame is not None else None
        if frame is None:
            continue
        results = model(frame, verbose=False, conf=YOLO_CONF, iou=YOLO_IOU)
        dets = []
        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name not in machine_class_set:
                    continue  # distractor hit (fruit/crate) -- not a machine, discard
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append((x1, y1, x2, y2, cls_name, float(box.conf[0])))
        dets.sort(key=lambda d: d[5], reverse=True)
        dets = dets[:MAX_DETECTIONS_SHOWN]
        with yolo_lock:
            yolo_result = dets


threading.Thread(target=yolo_worker, daemon=True).start()

person_model = YOLO("yolov8n.pt")
person_box = None           # largest person (x1, y1, x2, y2) in the latest frame
follow_active = False


def person_worker():
    global person_box
    while True:
        with yolo_lock:
            frame = yolo_frame.copy() if (yolo_frame is not None and follow_active) else None
        if frame is None:
            threading.Event().wait(0.05)
            continue
        r = person_model(frame, verbose=False, classes=[0], conf=0.3)[0]
        best = None
        for b in r.boxes.xyxy.cpu().numpy():
            if best is None or (b[3] - b[1]) > (best[3] - best[1]):
                best = tuple(int(v) for v in b)
        with yolo_lock:
            person_box = best


threading.Thread(target=person_worker, daemon=True).start()
follow_source = None

# Bridge to the separately-running Industrial Spare Parts Agentic System
# (FastAPI backend at :8000 - see that project's README for how to start
# it). Runs its own single-flight worker thread; maybe_trigger() below is
# the only call from the sim loop, and it's non-blocking.
agent_bridge = AgentBridge()
agent_bridge.start()

threading.Thread(target=lambda: MachineInfoPanel(agent_bridge).run(), daemon=True).start()

PID_CF = pid_velocity_fixed_height_controller()
past_x_global = 0
past_y_global = 0
past_time = robot.getTime()
height_desired = FLYING_ATTITUDE

print("=== Crazyflie + YOLO-World Machine Detection ===")
print("W/S = forward/back  |  A/D = strafe")
print("Up/Down arrows = altitude")
print("Q/E or Left/Right arrows = turn left/right (yaw) -- rotate in place to scan for more objects")
print("F = toggle autonomous FOLLOW of the factory worker")

while robot.step(timestep) != -1:
    dt = robot.getTime() - past_time
    if dt <= 0:
        dt = timestep / 1000.0

    # Sensor readings
    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    yaw = imu.getRollPitchYaw()[2]
    yaw_rate = gyro.getValues()[2]
    altitude = gps.getValues()[2]
    x_global = gps.getValues()[0]
    y_global = gps.getValues()[1]
    v_x_global = (x_global - past_x_global) / dt
    v_y_global = (y_global - past_y_global) / dt

    # Body-fixed velocities
    cosyaw = cos(yaw)
    sinyaw = sin(yaw)
    v_x = v_x_global * cosyaw + v_y_global * sinyaw
    v_y = -v_x_global * sinyaw + v_y_global * cosyaw

    forward_desired = 0
    sideways_desired = 0
    yaw_desired = 0
    height_diff_desired = 0

    key = keyboard.getKey()
    while key > 0:
        if key == ord('W'):
            forward_desired += 0.5
        elif key == ord('S'):
            forward_desired -= 0.5
        elif key == ord('A'):
            sideways_desired += 0.5
        elif key == ord('D'):
            sideways_desired -= 0.5
        elif key == Keyboard.UP:
            height_diff_desired = 0.1
        elif key == Keyboard.DOWN:
            height_diff_desired = -0.1
        elif key == Keyboard.LEFT or key == ord('Q'):
            yaw_desired = 1
        elif key == Keyboard.RIGHT or key == ord('E'):
            yaw_desired = -1
        elif key == ord('F'):
            follow_toggle = not follow_toggle
        key = keyboard.getKey()

    t_sim = robot.getTime()
    follow_active = follow_toggle or (AUTOPILOT and t_sim >= FOLLOW_START_T)
    if AUTOPILOT:
        mission_phase = ("TAKEOFF" if t_sim < 4 else "PHASE 1  SCAN" if t_sim < 17
                         else "PHASE 2  INSPECT MACHINE" if t_sim < FOLLOW_START_T
                         else "PHASE 3  FOLLOW WORKER")
    else:
        mission_phase = "FOLLOW WORKER" if follow_active else "MANUAL"
    if follow_active:
        with yolo_lock:
            pb = person_box
        cw = 640 if camera.getWidth() < 400 else camera.getWidth()
        ch = 480 if camera.getWidth() < 400 else camera.getHeight()
        forward_desired, sideways_desired, yaw_desired, follow_source = follow_command(
            pb, cw, ch, x_global, y_global, yaw)
    elif AUTOPILOT:
        follow_source = None
        forward_desired, sideways_desired, yaw_desired = autopilot_command(
            t_sim, x_global, y_global, yaw, scara_node)

    height_desired += height_diff_desired * dt

    motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                              yaw_desired, height_desired,
                              roll, pitch, yaw_rate,
                              altitude, v_x, v_y)

    m1_motor.setVelocity(-motor_power[0])
    m2_motor.setVelocity(motor_power[1])
    m3_motor.setVelocity(-motor_power[2])
    m4_motor.setVelocity(motor_power[3])

    past_time = robot.getTime()
    past_x_global = x_global
    past_y_global = y_global

    # Camera + YOLO-World overlay
    img = camera.getImage()
    if img:
        w = camera.getWidth()
        h = camera.getHeight()
        frame = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if w < 400:
            bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)

        with yolo_lock:
            yolo_frame = bgr.copy()
            dets = list(yolo_result)

        for (x1, y1, x2, y2, cls_name, conf) in dets:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 140, 255), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(bgr, label,
                        (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        gt_active = False
        if dets:
            best = max(dets, key=lambda d: d[5])
            bx1, by1, bx2, by2, blabel, bconf = best
            agent_bridge.maybe_trigger(bgr, (bx1, by1, bx2, by2), blabel, bconf)
        elif ground_truth_arm_visible(x_global, y_global, yaw, scara_node):
            # YOLO-World found nothing this frame, but the arm is genuinely
            # in view per real simulation geometry - use the full frame so
            # the pipeline still gets real pixels to work with, labeled
            # distinctly so it's never confused with an actual YOLO box.
            gt_active = True
            agent_bridge.maybe_trigger(bgr, (0, 0, bgr.shape[1], bgr.shape[0]),
                                        "GT: industrial machine (Epson SCARA T6)", 1.0)
            cv2.rectangle(bgr, (2, 2), (bgr.shape[1] - 3, bgr.shape[0] - 3), (255, 0, 200), 2)
            cv2.putText(bgr, "GROUND-TRUTH ASSIST ACTIVE (YOLO found nothing this frame)",
                        (10, bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)

        with yolo_lock:
            gt_detection_active = gt_active

        detected_now = bool(dets) or gt_active
        detection_source = "YOLO" if dets else ("GROUND-TRUTH" if gt_active else None)
        draw_agent_hud(bgr, agent_bridge, detected_now=detected_now, detection_source=detection_source)

        # HUD
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 55), (0, 0, 0), -1)
        cv2.putText(bgr, f"Pos  x:{x_global:.2f}  y:{y_global:.2f}  z:{altitude:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.putText(bgr, f"Machines detected: {len(dets)}   Yaw: {round(yaw, 2)}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if dets else (180, 180, 180), 1)

        if follow_active:
            with yolo_lock:
                pb = person_box
            col = (60, 60, 255)
            if pb is not None:
                x1, y1, x2, y2 = pb
                cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 3)
                cv2.putText(bgr, "LOCK worker", (x1, max(y1 - 8, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.line(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2), c, col, 2)
            cv2.drawMarker(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        banner = mission_phase + (f"   [{follow_source}]" if follow_active and follow_source else "")
        cv2.rectangle(bgr, (0, 55), (bgr.shape[1], 85), (40, 20, 0), -1)
        cv2.putText(bgr, banner, (10, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 220, 120) if follow_active else (255, 200, 80), 2)

        if show_window:
            try:
                cv2.imshow("Crazyflie - YOLO-World Machine Detection", bgr)
                cv2.waitKey(1)
            except cv2.error:
                # A headless OpenCV build has no GUI backend; keep flying
                # (and recording, if enabled) without the preview window.
                show_window = False
                print("cv2.imshow unavailable (headless OpenCV) - camera window disabled.")

        if RECORD_DIR and not recording_done:
            t_now = robot.getTime()
            if hud_writer is None:
                hud_writer = cv2.VideoWriter(os.path.join(RECORD_DIR, "hud.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS,
                                             (bgr.shape[1], bgr.shape[0]))
            while next_hud_t <= t_now:
                hud_writer.write(bgr)
                next_hud_t += 1.0 / RECORD_FPS
            if t_now >= RECORD_SECONDS:
                recording_done = True
                hud_writer.release()
                robot.movieStopRecording()
                print("Demo recording finished; waiting for Webots to encode the movie...")
                while not robot.movieIsReady() and robot.step(timestep) != -1:
                    pass
                print("DEMO_RECORDING_READY")
                robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

cv2.destroyAllWindows()
