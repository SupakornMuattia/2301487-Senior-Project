import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# การตั้งค่ากล้อง (Calibration สำหรับวัดระยะด้วยส่วนสูง)
# ─────────────────────────────────────────────
# หากระยะในโปรแกรมไม่ตรงกับตลับเมตร ให้ปรับตัวเลข 2 ตัวนี้ครับ
FOCAL_FACTOR_FULL_BODY = 1.4  # ตัวคูณสำหรับส่วนสูงเต็มตัว (จมูก-ข้อเท้า)
FOCAL_FACTOR_TORSO = 0.45  # ตัวคูณสำหรับความยาวลำตัว (ไหล่-สะโพก) ใช้ตอนขาหลุดเฟรม

# ─────────────────────────────────────────────
# MediaPipe Setup
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

# ─────────────────────────────────────────────
# สี UI และ AR Colors
# ─────────────────────────────────────────────
COLOR_BG = (15, 15, 25)
COLOR_TEAL = (0, 200, 170)
COLOR_CORAL = (60, 100, 230)
COLOR_AMBER = (30, 180, 230)
COLOR_WHITE = (240, 240, 240)
COLOR_GRAY = (100, 100, 120)
COLOR_GREEN = (50, 200, 100)
COLOR_RED = (60, 60, 220)
COLOR_GUIDE = (200, 200, 50)
COLOR_ORANGE = (30, 140, 255)

AR_COLORS = [
    ("สีแดง", (60, 60, 220)),
    ("สีเขียว", (50, 200, 100)),
    ("สีน้ำเงิน", (220, 100, 50)),
    ("สีเหลือง", (50, 200, 220)),
    ("สีม่วง", (200, 50, 200))
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# ค่าคงที่
# ─────────────────────────────────────────────
LOS_LATERAL = 8.0

RMS_WINDOW_SEC = 2.0
RMS_MAX = 0.12
RMS_HIP_WEIGHT = 0.6
RMS_SH_WEIGHT = 0.4

LOS_SCORE_WEIGHT = 0.8
RMS_SCORE_WEIGHT = 0.2

LEANING_DURATION_SEC = 60.0

# โหมด 2 Constants
TARGET_DISTANCE_MIN = 2.0

# โหมด 3 Constants
M3_DURATION_SEC = 60.0
M3_AR_TIMEOUT = 5.0
AR_COOLDOWN_SEC = 0.5

# ─────────────────────────────────────────────
# State Variables
# ─────────────────────────────────────────────
mode = 1
show_guide = True
current_instruction = None

# Mode 1: Leaning + AR
leaning_state = 0
ready_start_t = None
countdown_start_t = None
leaning_start = None
leaning_time = 0.0
ar_score = 0
ar_targets = []
final_score = None
final_ar_score = None

baseline_lean_lat = 0.0
lean_calibration_buf = []
arm_span_ratio = 0.35

# Cooldown State (สำหรับโหมด 3)
ar_cooldown_start = None
is_cooldown = False

# Mode 2 (Gait)
gait_state = 0
step_count = 0
step_count_steady = 0
current_tracking_part = "None"
current_gait_phase = "รอทดสอบ"
gait_start = None
gait_countdown_start = None
gait_time = 0.0
current_distance_m = 0.0
gait_start_distance_m = 0.0
walked_distance_m = 0.0

last_step_time = 0.0
step_y_history = deque(maxlen=15)
last_step_distance_m = 0.0
current_step_length = 0.0

# Mode 3 (Reach + Cognitive Color)
m3_state = 0
m3_countdown_start = None
m3_start_t = None
m3_time = 0.0
m3_targets = []
m3_spawn_t = 0.0
m3_hits = 0
m3_misses = 0
m3_rx_times = []
m3_bal_scores = []

m3_final_rx = 0.0
m3_final_acc = 0.0
m3_final_balance = 0.0
m3_final_items = 0

hip_buf = deque()
sh_buf = deque()

MODE_NAMES = {1: "โน้มตัวซ้าย/ขวา + หยิบของ (60s)",
              2: "ก้าวเดิน 2m (Gait Analysis)",
              3: "Dynamic Reach & Cognitive AR"}


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def lm_px(lm, w, h): return int(lm.x * w), int(lm.y * h)


# 🚀 อัปเดต: ระบบวัดระยะด้วย "ส่วนสูงเต็มตัว" และ "ความยาวลำตัว" แทนความกว้างไหล่
def estimate_distance(lms):
    nose = lms[mp_pose.PoseLandmark.NOSE]
    la = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # กรณี 1: มองเห็นข้อเท้า (วัดส่วนสูงจากจมูกถึงข้อเท้า)
    if la.visibility > 0.5 and ra.visibility > 0.5:
        avg_ankle_y = (la.y + ra.y) / 2.0
        body_height_norm = abs(avg_ankle_y - nose.y)

        if body_height_norm < 0.01: return 10.0
        return min(FOCAL_FACTOR_FULL_BODY / body_height_norm, 10.0)

    # กรณี 2: มองไม่เห็นข้อเท้า (วัดความยาวลำตัวจากไหล่ถึงสะโพกแทน)
    else:
        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]

        avg_sh_y = (ls.y + rs.y) / 2.0
        avg_hip_y = (lh.y + rh.y) / 2.0
        torso_height_norm = abs(avg_hip_y - avg_sh_y)

        if torso_height_norm < 0.01: return 10.0
        return min(FOCAL_FACTOR_TORSO / torso_height_norm, 10.0)


def calc_lean_lat(lms):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER];
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP];
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    sh = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2])
    hp = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2, (lh.z + rh.z) / 2])
    d = sh - hp
    return float(np.degrees(np.arctan2(d[0], abs(d[1]) + 1e-6)))


def update_sway_buffers(lms, now):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER];
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP];
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_buf.append(((lh.x + rh.x) / 2, (lh.y + rh.y) / 2, now))
    sh_buf.append(((ls.x + rs.x) / 2, (ls.y + rs.y) / 2, now))
    cutoff = now - RMS_WINDOW_SEC
    while hip_buf and hip_buf[0][2] < cutoff: hip_buf.popleft()
    while sh_buf and sh_buf[0][2] < cutoff: sh_buf.popleft()


def calc_rms_sway():
    if len(hip_buf) < 5: return 0.0, 0.0, 0.0
    ha = np.array([(x, y) for x, y, _ in hip_buf]);
    sa = np.array([(x, y) for x, y, _ in sh_buf])
    rms_h = float(np.sqrt(np.mean((ha[:, 0] - ha[:, 0].mean()) ** 2 + (ha[:, 1] - ha[:, 1].mean()) ** 2)))
    rms_s = float(np.sqrt(np.mean((sa[:, 0] - sa[:, 0].mean()) ** 2 + (sa[:, 1] - sa[:, 1].mean()) ** 2)))
    return rms_h, rms_s, rms_h * RMS_HIP_WEIGHT + rms_s * RMS_SH_WEIGHT


def calc_los_penalty(lat):
    over_lat = max(0.0, abs(lat) - LOS_LATERAL)
    return min(100.0, (over_lat / LOS_LATERAL) * 100)


def calc_rms_penalty(rms_combined): return min(100.0, (rms_combined / RMS_MAX) * 100)


def calc_balance_score(lat, rms_combined):
    p = calc_los_penalty(lat) * LOS_SCORE_WEIGHT + calc_rms_penalty(rms_combined) * RMS_SCORE_WEIGHT
    return max(0, round(100 - p))


def score_color(
        s): return COLOR_GREEN if s >= 80 else COLOR_AMBER if s >= 60 else COLOR_ORANGE if s >= 40 else COLOR_RED


def score_label(s): return "ดี" if s >= 80 else "ปานกลาง" if s >= 60 else "เสี่ยง" if s >= 40 else "เสี่ยงสูง"


def analyze_gait_phases_2m(lms):
    return "กำลังทดสอบ 2 เมตร (นับรวมทุกก้าว)"


def detect_step_with_fallback(lms, now):
    global current_tracking_part, last_step_time, step_y_history

    la, ra = lms[mp_pose.PoseLandmark.LEFT_ANKLE], lms[mp_pose.PoseLandmark.RIGHT_ANKLE]
    lk, rk = lms[mp_pose.PoseLandmark.LEFT_KNEE], lms[mp_pose.PoseLandmark.RIGHT_KNEE]

    is_step = False

    def is_valid(lm):
        return lm.visibility > 0.5 and lm.y < 0.95

    # ตัด Hip ออกตามความต้องการ ใช้แค่ข้อเท้าและหัวเข่า
    if is_valid(la) and is_valid(ra):
        part = "Ankle (ข้อเท้า)"
        avg_y = (la.y + ra.y) / 2.0
        step_threshold = 0.020
    elif is_valid(lk) and is_valid(rk):
        part = "Knee (หัวเข่า)"
        avg_y = (lk.y + rk.y) / 2.0
        step_threshold = 0.010
    else:
        part = "None"
        return part, False

    if current_tracking_part != part:
        current_tracking_part = part
        step_y_history.clear()
        return part, False

    step_y_history.append((avg_y, now))

    if len(step_y_history) >= 15:
        y_vals = [y for y, t in step_y_history]
        baseline = sum(y_vals) / len(y_vals)
        oscillation = abs(avg_y - baseline)

        if oscillation > step_threshold:
            if now - last_step_time > 0.6:
                is_step = True
                last_step_time = now
                step_y_history.clear()

    return part, is_step


def calc_trunk_sway_simple(lms):
    return abs(((lms[11].x + lms[12].x) / 2) - ((lms[23].x + lms[24].x) / 2)) * 100


# ─────────────────────────────────────────────
# AR Target Functions
# ─────────────────────────────────────────────
def get_user_center(lms):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    return (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0


def spawn_target_mode1(w, h, user_cx, user_cy, arm_span, existing_targets=None):
    min_dist_sq = 0.15 ** 2
    arm_span = max(0.35, arm_span)

    for _ in range(100):
        zone = np.random.choice(['top', 'mid_left', 'mid_right'])

        if zone == 'top':
            angle = np.random.uniform(np.pi / 6, 5 * np.pi / 6)
        elif zone == 'mid_left':
            angle = np.random.uniform(5 * np.pi / 6, 7 * np.pi / 6)
        else:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)

        r = arm_span * np.random.uniform(0.70, 0.90)
        tx = user_cx + r * np.cos(angle)
        ty = user_cy - r * np.sin(angle)

        tx = float(np.clip(tx, 0.1, 0.9))
        ty = float(np.clip(ty, 0.15, 0.8))

        if not existing_targets:
            break

        too_close = any((tx - ex_x) ** 2 + (ty - ex_y) ** 2 < min_dist_sq for ex_x, ex_y in existing_targets)
        if not too_close:
            break

    return (tx, ty)


def spawn_target_dynamic(w, h, user_cx, user_cy, arm_span, existing_targets=None, has_color=False):
    min_dist_sq = 0.15 ** 2
    arm_span = max(0.35, arm_span)

    for _ in range(100):
        angle = np.random.uniform(0, np.pi)
        r = arm_span * np.random.uniform(0.8, 1.0)

        tx = user_cx + r * np.cos(angle)
        ty = user_cy - r * np.sin(angle)

        tx = float(np.clip(tx, 0.1, 0.9))
        ty = float(np.clip(ty, 0.15, 0.8))

        dist_to_center = np.hypot(tx - user_cx, ty - user_cy)
        if dist_to_center < (arm_span * 0.5):
            continue

        if not existing_targets:
            break

        if has_color:
            too_close = any((tx - ex_x) ** 2 + (ty - ex_y) ** 2 < min_dist_sq for ex_x, ex_y, _, _ in existing_targets)
        else:
            too_close = any((tx - ex_x) ** 2 + (ty - ex_y) ** 2 < min_dist_sq for ex_x, ex_y in existing_targets)

        if not too_close:
            break

    if has_color:
        cname, cbgr = AR_COLORS[np.random.randint(len(AR_COLORS))]
        return (tx, ty, cname, cbgr)
    return (tx, ty)


def reset_ar_task_dynamic_color(w, h, user_cx, user_cy, arm_span):
    inst_color = AR_COLORS[np.random.randint(len(AR_COLORS))]
    first_target = spawn_target_dynamic(w, h, user_cx, user_cy, arm_span, existing_targets=[], has_color=True)
    first_target = (first_target[0], first_target[1], inst_color[0], inst_color[1])
    targets = [first_target]

    distractor_colors = [c for c in AR_COLORS if c[0] != inst_color[0]]
    for _ in range(2):
        dist_color = distractor_colors[np.random.randint(len(distractor_colors))]
        t = spawn_target_dynamic(w, h, user_cx, user_cy, arm_span, existing_targets=targets, has_color=True)
        t = (t[0], t[1], dist_color[0], dist_color[1])
        targets.append(t)
    np.random.shuffle(targets)
    return inst_color, targets


def check_reach_basic(lms, targets, w, h):
    lw, rw = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    hits = []
    for i, (tx, ty) in enumerate(targets):
        hit_left = lw.visibility > 0.6 and np.hypot(lw.x - tx, lw.y - ty) < 0.08
        hit_right = rw.visibility > 0.6 and np.hypot(rw.x - tx, rw.y - ty) < 0.08
        if hit_left or hit_right:
            hits.append(i)
    return hits


def check_reach_color(lms, targets, w, h):
    lw, rw = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    hits = []
    for i, (tx, ty, cname, cbgr) in enumerate(targets):
        hit_left = lw.visibility > 0.6 and np.hypot(lw.x - tx, lw.y - ty) < 0.08
        hit_right = rw.visibility > 0.6 and np.hypot(rw.x - tx, rw.y - ty) < 0.08
        if hit_left or hit_right:
            hits.append(i)
    return hits


def draw_reach_targets_basic(frame, targets, w, h):
    for (tx, ty) in targets:
        px, py = int(tx * w), int(ty * h)
        cv2.circle(frame, (px, py), 26, COLOR_CORAL, -1)
        cv2.circle(frame, (px, py), 26, COLOR_WHITE, 2)
        draw_text_bold(frame, "AR", (px - 13, py + 7), 0.5, COLOR_WHITE)


def draw_reach_targets_color(frame, targets, w, h):
    for (tx, ty, cname, cbgr) in targets:
        px, py = int(tx * w), int(ty * h)
        cv2.circle(frame, (px, py), 26, cbgr, -1)
        cv2.circle(frame, (px, py), 26, COLOR_WHITE, 2)


# ─────────────────────────────────────────────
# UI Drawing
# ─────────────────────────────────────────────
def put_thai_text(frame, text, pos, font_size=20, color=(240, 240, 240)):
    b, g, r = color
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drw = ImageDraw.Draw(pil)
    try:
        fnt = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        fnt = ImageFont.load_default()
    drw.text(pos, text, font=fnt, fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)[:]


def draw_text_th(frame, text, pos, scale=0.55, color=COLOR_WHITE, **_): put_thai_text(frame, text, pos, int(scale * 40),
                                                                                      color)


def draw_text_bold(frame, text, pos, scale=0.7, color=COLOR_WHITE, **_): put_thai_text(frame, text, pos,
                                                                                       int(scale * 45), color)


def draw_panel(frame, x, y, w, h, color, alpha=0.35):
    ov = frame.copy();
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame);
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


def draw_bar(frame, x, y, value, max_val, bw, bh, color):
    fill = int(np.clip(value / max_val * bw, 0, bw))
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_GRAY, -1)
    if fill > 0: cv2.rectangle(frame, (x, y), (x + fill, y + bh), color, -1)


def draw_limit_marker(frame, x, y, bw, bh):
    cv2.line(frame, (x, y - 2), (x, y + bh + 2), COLOR_WHITE, 2)


def draw_score_ring(frame, cx, cy, score, r=28):
    color = score_color(score);
    cv2.circle(frame, (cx, cy), r, COLOR_GRAY, 3)
    for a in range(0, int(360 * score / 100), 3):
        rad = np.radians(a - 90);
        cv2.circle(frame, (int(cx + r * np.cos(rad)), int(cy + r * np.sin(rad))), 2, color, -1)
    cv2.putText(frame, str(score), (cx - (14 if score >= 100 else 10 if score >= 10 else 6), cy + 5), FONT_BOLD, 0.55,
                color, 1, cv2.LINE_AA)


def draw_guide(frame, mode, w, h):
    if mode == 1:
        cx = w // 2;
        cv2.line(frame, (cx, 50), (cx, h - 50), COLOR_GUIDE, 1, cv2.LINE_AA)
        for deg in [-LOS_LATERAL, LOS_LATERAL]:
            off = int(np.tan(np.radians(deg)) * (h // 2));
            cv2.line(frame, (cx, h // 2), (cx + off, h - 50), COLOR_GUIDE, 1, cv2.LINE_AA)
    elif mode == 2:
        cx = w // 2
        for off in [-40, 40]: cv2.line(frame, (cx + off, 80), (cx + off, h - 80), COLOR_GUIDE, 1, cv2.LINE_AA)


def draw_info_panel(frame, mode, lean_lat, balance_score, hip_rms, w, h, **kwargs):
    pw = 290
    if mode == 1:
        ph = 300
    elif mode == 2:
        ph = 290
    elif mode == 3:
        ph = 350 if m3_state == 3 else 250
    else:
        ph = 200

    draw_panel(frame, 10, 10, pw, ph, COLOR_BG, 0.82)
    draw_text_bold(frame, f"Mode: {mode}", (20, 38), 0.55, COLOR_TEAL)
    draw_text_th(frame, MODE_NAMES[mode], (20, 62), 0.5, COLOR_WHITE)
    y = 90

    if mode == 1:
        if leaning_state == 1:
            draw_text_th(frame, "กำลังเซ็ตศูนย์...", (20, y), color=COLOR_AMBER)
        elif leaning_state == 2:
            draw_text_th(frame, "รอวัดแขน...", (20, y), color=COLOR_ORANGE)
        elif leaning_state == 3:
            draw_text_th(frame, "นับถอยหลัง...", (20, y), color=COLOR_AMBER)
        elif leaning_state == 4:
            rem = max(0.0, LEANING_DURATION_SEC - leaning_time)
            draw_text_th(frame, f"ทดสอบ: {rem:.1f} วินาที", (20, y), color=COLOR_AMBER if rem > 10 else COLOR_RED)
        elif final_score is not None:
            draw_text_th(frame, "เสร็จสิ้น", (20, y), color=COLOR_GREEN)
        else:
            draw_text_th(frame, "กด [SPACE] เริ่ม 60s", (20, y), color=COLOR_GRAY)

        display_score = final_score if final_score is not None else balance_score
        display_ar = final_ar_score if final_ar_score is not None else ar_score

        y += 24;
        draw_text_th(frame, f"AR หยิบได้: {display_ar} ครั้ง", (20, y), color=COLOR_TEAL)
        y += 26;
        BW = 245;
        abs_lat = abs(lean_lat)
        col_lat = COLOR_RED if abs_lat > LOS_LATERAL else COLOR_GREEN
        dir_lat = "ขวา" if lean_lat > 0 else "ซ้าย" if lean_lat < 0 else "ตรง"
        draw_text_th(frame, f"เอียง: {lean_lat:+.1f}° ({dir_lat}) limit ±{LOS_LATERAL}°", (20, y), color=col_lat)
        y += 19;
        draw_bar(frame, 20, y, abs_lat, LOS_LATERAL * 1.5, BW, 9, col_lat)
        draw_limit_marker(frame, 20 + int(BW / 1.5), y, BW, 9)
        y += 24;
        rms_col = COLOR_GREEN if hip_rms < RMS_MAX * 0.5 else COLOR_AMBER if hip_rms < RMS_MAX else COLOR_RED
        draw_text_th(frame, f"RMS Hip:{hip_rms:.4f}", (20, y), 0.44, rms_col)
        y += 18;
        draw_bar(frame, 20, y, hip_rms, RMS_MAX, BW, 9, rms_col)
        draw_limit_marker(frame, 20 + BW, y, BW, 9)
        y += 24;
        sc = score_color(display_score)
        draw_score_ring(frame, 48, y + 25, display_score)
        draw_text_bold(frame, f"คะแนน: {display_score}/100", (90, y + 16), 0.55, sc)
        draw_text_th(frame, f"ระดับ: {score_label(display_score)}", (90, y + 40), color=sc)

    elif mode == 2:
        gait_time = kwargs.get('gait_time', 0)
        step_count = kwargs.get('step_count', 0)
        walked_dist = kwargs.get('walked_dist', 0.0)
        step_length = kwargs.get('step_length', 0.0)
        simple_sway = kwargs.get('simple_sway', 0)
        current_gait_phase = kwargs.get('current_gait_phase', 'รอเริ่ม')
        current_tracking_part = kwargs.get('current_tracking_part', 'None')
        current_dist_m = kwargs.get('current_dist_m', 0.0)
        g_state = kwargs.get('gait_state', 0)

        if g_state == 1:
            draw_text_th(frame, f"จัดตำแหน่ง: ห่าง {current_dist_m:.1f} ม.", (20, y), color=COLOR_ORANGE);
            y += 26
        elif g_state == 2:
            rem_cd = max(1, int(np.ceil(3.0 - (time.time() - kwargs.get('gait_cd', 0)))))
            draw_text_th(frame, f"เตรียบเดิน: {rem_cd} วิ", (20, y), color=COLOR_AMBER);
            y += 26
        else:
            timer_str = f"{gait_time:.1f} วินาที" if g_state == 3 or gait_time > 0 else "กด [SPACE] จัดระยะ"
            draw_text_th(frame, f"เวลา: {timer_str}", (20, y));
            y += 26

        draw_text_th(frame, f"ก้าวรวม: {step_count} ก้าว", (20, y), color=COLOR_AMBER);
        y += 26
        draw_text_th(frame, f"ระยะที่เดินแล้ว: {walked_dist:.1f} ม.", (20, y), color=COLOR_GREEN);
        y += 26
        draw_text_th(frame, f"ระยะก้าวล่าสุด: {step_length:.2f} ม.", (20, y), color=COLOR_WHITE);
        y += 26
        draw_text_th(frame, f"Trunk sway: {simple_sway:.1f}%", (20, y), color=COLOR_TEAL);
        y += 26
        draw_text_th(frame, f"จุดอ้างอิง: {current_tracking_part}", (20, y), color=COLOR_ORANGE);
        y += 26

        status = "กำลังจัดระยะ" if g_state == 1 else "กำลังทดสอบ..." if g_state == 3 else "เสร็จสิ้น" if gait_time > 0 else "พร้อม"
        draw_text_th(frame, f"สถานะ: {status}", (20, y))

    elif mode == 3:
        if m3_state == 1:
            draw_text_th(frame, f"รอเริ่มทดสอบ...", (20, y), color=COLOR_AMBER)
        elif m3_state == 2:
            rem = max(0.0, M3_DURATION_SEC - m3_time)
            draw_text_th(frame, f"เหลือเวลา: {rem:.1f} วินาที", (20, y), color=COLOR_AMBER if rem > 10 else COLOR_RED);
            y += 30
            draw_text_th(frame, f"1. ตอบสนอง: (กำลังวัด...)", (20, y));
            y += 24
            draw_text_th(frame, f"2. แม่นยำ: (กำลังวัด...)", (20, y));
            y += 24
            draw_text_th(frame, f"3. ทรงตัวเฉลี่ย: {balance_score}/100", (20, y), color=COLOR_TEAL);
            y += 24
            draw_text_th(frame, f"4. หยิบถูก: {m3_hits} ครั้ง", (20, y), color=COLOR_GREEN)
        elif m3_state == 3:
            draw_text_th(frame, "ผลการทดสอบ (60 วินาที):", (20, y), color=COLOR_GREEN);
            y += 30
            draw_text_th(frame, f"1. ตอบสนองเฉลี่ย: {m3_final_rx:.2f} วิ", (20, y), color=COLOR_WHITE);
            y += 24
            draw_text_th(frame, f"2. แม่นยำ (ถูก/ทั้งหมด): {m3_final_acc:.1f}%", (20, y), color=COLOR_WHITE);
            y += 24
            draw_text_th(frame, f"3. ทรงตัวขณะเคลื่อนไหว: {int(m3_final_balance)}/100", (20, y), color=COLOR_WHITE);
            y += 24
            draw_text_th(frame, f"4. หยิบถูกรวม: {m3_final_items} ครั้ง/นาที", (20, y), color=COLOR_TEAL);
            y += 30
            draw_text_th(frame, "กด [SPACE] ทดสอบใหม่", (20, y), 0.45, COLOR_GRAY)
        else:
            draw_text_th(frame, "กด [SPACE] เริ่มทดสอบ 60 วินาที", (20, y), color=COLOR_GRAY)


def draw_controls(frame, w):
    items = [("[1] โน้มตัว+หยิบ", mode == 1), ("[2] ก้าวเดิน", mode == 2),
             ("[3] เอื้อมมือ+สี", mode == 3), ("[G] Guide", show_guide),
             ("[SPACE] Action", False), ("[Q] ออก", False)]
    px = w - 195
    draw_panel(frame, px - 5, 10, 190, len(items) * 26 + 16, COLOR_BG, 0.80)
    for i, (lbl, active) in enumerate(items):
        draw_text_th(frame, lbl, (px, 36 + i * 26), color=COLOR_TEAL if active else COLOR_GRAY)


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
def main():
    global mode, show_guide, current_instruction

    global leaning_state, ready_start_t, countdown_start_t, leaning_start, leaning_time, ar_score, ar_targets, final_score, final_ar_score
    global baseline_lean_lat, lean_calibration_buf, arm_span_ratio

    global is_cooldown, ar_cooldown_start

    global gait_state, gait_start, gait_countdown_start, gait_time, step_count, step_count_steady
    global current_tracking_part, current_gait_phase, current_distance_m, last_step_time, step_y_history
    global gait_start_distance_m, walked_distance_m, last_step_distance_m, current_step_length

    global m3_state, m3_countdown_start, m3_start_t, m3_time, m3_targets, m3_spawn_t, m3_hits, m3_misses, m3_rx_times, m3_bal_scores
    global m3_final_rx, m3_final_acc, m3_final_balance, m3_final_items

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    lean_lat = balance_score = hip_rms = simple_sway = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 📱 ครอปเป็นสัดส่วนหน้าจอมือถือ (9:16)
        h_orig, w_orig = frame.shape[:2]
        target_aspect = 9 / 16
        current_aspect = w_orig / h_orig

        if current_aspect > target_aspect:
            new_w = int(h_orig * target_aspect)
            start_x = (w_orig - new_w) // 2
            frame = frame[:, start_x:start_x + new_w]
        else:
            new_h = int(w_orig / target_aspect)
            start_y = (h_orig - new_h) // 2
            frame = frame[start_y:start_y + new_h, :]

        frame = cv2.resize(frame, (540, 960))
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        if show_guide: draw_guide(frame, mode, w, h)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=COLOR_TEAL, thickness=4, circle_radius=4),
                                   mp_draw.DrawingSpec(color=COLOR_WHITE, thickness=2))

            raw_lean_lat = calc_lean_lat(lms)
            update_sway_buffers(lms, now)
            hip_rms, sh_rms, rms_comb = calc_rms_sway()

            if is_cooldown and now - ar_cooldown_start >= AR_COOLDOWN_SEC:
                is_cooldown = False

            if mode == 1:
                if leaning_state == 0 or leaning_state == 1 or leaning_state == 2 or final_score is not None:
                    lean_lat = 0.0
                else:
                    lean_lat = raw_lean_lat - baseline_lean_lat

                current_score = calc_balance_score(lean_lat, rms_comb)

                if leaning_state == 1:
                    lean_calibration_buf.append(raw_lean_lat)
                    if hip_rms < RMS_MAX * 0.8:
                        if ready_start_t is None:
                            ready_start_t = now
                        elif now - ready_start_t >= 1.0:
                            leaning_state = 2;
                            ready_start_t = None
                            baseline_lean_lat = np.mean(lean_calibration_buf) if lean_calibration_buf else raw_lean_lat
                    else:
                        ready_start_t = None
                        lean_calibration_buf.clear()
                    balance_score = 100

                elif leaning_state == 2:
                    lw = lms[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
                    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    arm_span = abs(lw.x - rw.x)
                    shoulder_w = abs(ls.x - rs.x)

                    if lw.visibility > 0.5 and rw.visibility > 0.5 and arm_span > shoulder_w * 2.5:
                        if ready_start_t is None:
                            ready_start_t = now
                        elif now - ready_start_t >= 1.0:
                            arm_span_ratio = arm_span / 2.0
                            leaning_state = 3;
                            countdown_start_t = now
                    else:
                        ready_start_t = None
                    balance_score = 100

                elif leaning_state == 3:
                    if countdown_start_t is not None and now - countdown_start_t >= 3.0:
                        leaning_state = 4;
                        leaning_start = now;
                        leaning_time = 0.0;
                        ar_score = 0
                        ar_targets = []
                        user_cx, user_cy = get_user_center(lms)
                        for _ in range(3):
                            ar_targets.append(
                                spawn_target_mode1(w, h, user_cx, user_cy, arm_span_ratio, existing_targets=ar_targets))
                        hip_buf.clear();
                        sh_buf.clear()
                    balance_score = current_score

                elif leaning_state == 4:
                    leaning_time = now - leaning_start
                    balance_score = current_score

                    hits = check_reach_basic(lms, ar_targets, w, h)
                    if hits:
                        ar_score += len(hits)
                        ar_targets = [t for i, t in enumerate(ar_targets) if i not in hits]
                        user_cx, user_cy = get_user_center(lms)
                        while len(ar_targets) < 3:
                            ar_targets.append(
                                spawn_target_mode1(w, h, user_cx, user_cy, arm_span_ratio, existing_targets=ar_targets))

                    draw_reach_targets_basic(frame, ar_targets, w, h)

                    for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        if lms[lm_id].visibility > 0.6:
                            px, py = lm_px(lms[lm_id], w, h);
                            cv2.circle(frame, (px, py), 12, COLOR_CORAL, -1)

                    if leaning_time >= LEANING_DURATION_SEC:
                        leaning_state = 0;
                        final_score = balance_score;
                        final_ar_score = ar_score
                        ar_targets = []
                else:
                    balance_score = 100

                ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER];
                rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lms[mp_pose.PoseLandmark.LEFT_HIP];
                rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
                sh_mid = (int((ls.x + rs.x) / 2 * w), int((ls.y + rs.y) / 2 * h))
                hp_mid = (int((lh.x + rh.x) / 2 * w), int((lh.y + rh.y) / 2 * h))
                exceed = abs(lean_lat) > LOS_LATERAL
                lc = COLOR_RED if exceed else COLOR_GREEN
                cv2.line(frame, sh_mid, hp_mid, lc, 3, cv2.LINE_AA)
                cv2.circle(frame, sh_mid, 8, lc, -1)
                cv2.circle(frame, hp_mid, 8, lc, -1)

            elif mode == 2:
                current_distance_m = estimate_distance(lms)

                if gait_state == 1:
                    cv2.rectangle(frame, (w // 2 - 250, 10), (w // 2 + 250, 90), COLOR_BG, -1)
                    dist_color = COLOR_GREEN if current_distance_m >= TARGET_DISTANCE_MIN else COLOR_ORANGE
                    draw_text_bold(frame, f"ระยะห่าง: {current_distance_m:.1f} เมตร", (w // 2 - 150, 30), 0.8,
                                   dist_color)

                    if current_distance_m < TARGET_DISTANCE_MIN:
                        draw_text_bold(frame, f"(กรุณาถอยหลังไปที่ระยะ {TARGET_DISTANCE_MIN} เมตร)", (w // 2 - 190, 70),
                                       0.6, COLOR_GRAY)
                    else:
                        draw_text_bold(frame, "(ระยะเหมาะสม!)", (w // 2 - 100, 70), 0.6, COLOR_GREEN)
                        gait_state = 2;
                        gait_countdown_start = now

                elif gait_state == 2:
                    rem_cd = max(1, int(np.ceil(3.0 - (now - gait_countdown_start))))
                    cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
                    draw_text_bold(frame, f"เริ่มเดินใน... {rem_cd}", (w // 2 - 110, 30), 0.8, COLOR_ORANGE)
                    if now - gait_countdown_start >= 3.0:
                        gait_state = 3;
                        gait_start = now;
                        gait_time = 0.0
                        step_count = 0;
                        step_count_steady = 0
                        gait_start_distance_m = current_distance_m
                        walked_distance_m = 0.0
                        last_step_distance_m = current_distance_m
                        current_step_length = 0.0

                elif gait_state == 3:
                    current_gait_phase = analyze_gait_phases_2m(lms)
                    simple_sway = calc_trunk_sway_simple(lms)
                    gait_time = now - gait_start
                    walked_distance_m = max(0.0, gait_start_distance_m - current_distance_m)

                    part_name, is_step = detect_step_with_fallback(lms, now)
                    current_tracking_part = part_name

                    if is_step:
                        step_count += 1
                        current_step_length = abs(last_step_distance_m - current_distance_m)
                        last_step_distance_m = current_distance_m

                    if "Ankle" in current_tracking_part:
                        pts = [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
                    elif "Knee" in current_tracking_part:
                        pts = [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]
                    else:
                        pts = []

                    for lm_id in pts:
                        px, py = lm_px(lms[lm_id], w, h);
                        cv2.circle(frame, (px, py), 14, COLOR_AMBER, -1)

            elif mode == 3:
                lean_lat = raw_lean_lat - baseline_lean_lat if m3_state == 2 else 0.0
                current_score = calc_balance_score(lean_lat, rms_comb)
                balance_score = current_score

                if m3_state == 1:
                    lean_calibration_buf.append(raw_lean_lat)
                    if now - m3_countdown_start >= 3.0:
                        m3_state = 2;
                        m3_start_t = now;
                        m3_time = 0.0
                        m3_hits = 0;
                        m3_misses = 0;
                        m3_rx_times = [];
                        m3_bal_scores = []
                        is_cooldown = False

                        user_cx, user_cy = get_user_center(lms)
                        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER];
                        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        arm_span_ratio = abs(ls.x - rs.x) * 2.0

                        current_instruction, m3_targets = reset_ar_task_dynamic_color(w, h, user_cx, user_cy,
                                                                                      arm_span_ratio)
                        m3_spawn_t = now
                        baseline_lean_lat = np.mean(lean_calibration_buf) if lean_calibration_buf else raw_lean_lat
                        hip_buf.clear();
                        sh_buf.clear();
                        lean_calibration_buf.clear()

                elif m3_state == 2:
                    m3_time = now - m3_start_t
                    m3_bal_scores.append(current_score)
                    user_cx, user_cy = get_user_center(lms)

                    if not is_cooldown:
                        hits = check_reach_color(lms, m3_targets, w, h)
                        if hits:
                            hit_idx = hits[0]
                            if m3_targets[hit_idx][2] == current_instruction[0]:
                                m3_hits += 1;
                                m3_rx_times.append(now - m3_spawn_t);
                                m3_targets = []
                                is_cooldown = True;
                                ar_cooldown_start = now
                                current_instruction, m3_targets = reset_ar_task_dynamic_color(w, h, user_cx, user_cy,
                                                                                              arm_span_ratio)
                                m3_spawn_t = now + AR_COOLDOWN_SEC
                            else:
                                m3_misses += 1;
                                m3_targets.pop(hit_idx)
                                dist_color = [c for c in AR_COLORS if c[0] != current_instruction[0]][
                                    np.random.randint(4)]
                                m3_targets.append(spawn_target_dynamic(w, h, user_cx, user_cy, arm_span_ratio,
                                                                       existing_targets=m3_targets, has_color=True))

                        elif now - m3_spawn_t > M3_AR_TIMEOUT:
                            m3_misses += 1;
                            m3_targets = [];
                            is_cooldown = True;
                            ar_cooldown_start = now
                            current_instruction, m3_targets = reset_ar_task_dynamic_color(w, h, user_cx, user_cy,
                                                                                          arm_span_ratio)
                            m3_spawn_t = now + AR_COOLDOWN_SEC

                    if not is_cooldown: draw_reach_targets_color(frame, m3_targets, w, h)
                    for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        if lms[lm_id].visibility > 0.6:
                            px, py = lm_px(lms[lm_id], w, h);
                            cv2.circle(frame, (px, py), 12, COLOR_CORAL, -1)

                    if m3_time >= M3_DURATION_SEC:
                        m3_state = 3;
                        m3_targets = [];
                        current_instruction = None;
                        is_cooldown = False
                        m3_final_rx = np.mean(m3_rx_times) if len(m3_rx_times) > 0 else 0.0
                        total_spawned = m3_hits + m3_misses
                        m3_final_acc = (m3_hits / total_spawned * 100) if total_spawned > 0 else 0.0
                        m3_final_balance = np.mean(m3_bal_scores) if len(m3_bal_scores) > 0 else 0.0
                        m3_final_items = m3_hits

        else:
            draw_text_bold(frame, "ไม่พบผู้ใช้", (w // 2 - 100, h // 2), color=COLOR_RED)
            hip_buf.clear();
            sh_buf.clear()

        # แจ้งเตือนกลางหน้าจอ โหมด 1
        if mode == 1:
            if leaning_state == 1:
                cv2.rectangle(frame, (w // 2 - 200, 30), (w // 2 + 200, 100), COLOR_BG, -1)
                draw_text_bold(frame, "ยืนนิ่งๆ เพื่อเซ็ตศูนย์กลาง...", (w // 2 - 170, 50), 0.8, COLOR_AMBER)
            elif leaning_state == 2:
                cv2.rectangle(frame, (w // 2 - 250, 30), (w // 2 + 250, 100), COLOR_BG, -1)
                if ready_start_t is not None:
                    rem_hold = max(0.0, 1.0 - (now - ready_start_t))
                    draw_text_bold(frame, f"ค้างไว้... {rem_hold:.1f} วิ", (w // 2 - 100, 50), 0.8, COLOR_GREEN)
                else:
                    draw_text_bold(frame, "กางแขนออกด้านข้าง (T-Pose)", (w // 2 - 200, 50), 0.8, COLOR_ORANGE)
            elif leaning_state == 3:
                rem_cd = max(1, int(np.ceil(3.0 - (now - countdown_start_t))))
                cv2.rectangle(frame, (w // 2 - 150, 30), (w // 2 + 150, 100), COLOR_BG, -1)
                draw_text_bold(frame, f"เริ่มใน... {rem_cd}", (w // 2 - 90, 50), 1.0, COLOR_GREEN)

        # แจ้งเตือนกลางหน้าจอ โหมด 3
        elif mode == 3:
            if m3_state == 1:
                rem_cd = max(1, int(np.ceil(3.0 - (now - m3_countdown_start))))
                cv2.rectangle(frame, (w // 2 - 150, 30), (w // 2 + 150, 100), COLOR_BG, -1)
                draw_text_bold(frame, f"เริ่มใน... {rem_cd}", (w // 2 - 90, 50), 1.0, COLOR_GREEN)
            elif m3_state == 2:
                if current_instruction:
                    if is_cooldown:
                        cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
                        draw_text_bold(frame, "ดึงมือกลับ...", (w // 2 - 80, 30), 0.8, COLOR_GRAY)
                    else:
                        cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
                        draw_text_bold(frame, f"คำสั่ง: แตะเป้าหมาย {current_instruction[0]}", (w // 2 - 160, 30), 0.8,
                                       current_instruction[1])
                        rem_obj = max(0, M3_AR_TIMEOUT - (now - m3_spawn_t))
                        cv2.rectangle(frame, (w // 2 - 170, 70),
                                      (w // 2 - 170 + int(340 * (rem_obj / M3_AR_TIMEOUT)), 75), current_instruction[1],
                                      -1)

        is_playing = False
        if mode == 1 and leaning_state in [1, 2, 3, 4]:
            is_playing = True
        elif mode == 2 and gait_state in [1, 2, 3]:
            is_playing = True
        elif mode == 3 and m3_state in [1, 2]:
            is_playing = True

        if not is_playing:
            draw_info_panel(frame, mode, lean_lat, balance_score, hip_rms, w=w, h=h,
                            gait_time=gait_time, step_count=step_count, step_count_steady=step_count_steady,
                            simple_sway=simple_sway, current_gait_phase=current_gait_phase,
                            current_tracking_part=current_tracking_part, current_dist_m=current_distance_m,
                            gait_state=gait_state, gait_cd=gait_countdown_start, walked_dist=walked_distance_m,
                            step_length=current_step_length)
            draw_controls(frame, w)
        else:
            if mode == 1 and leaning_state == 4:
                rem = max(0.0, LEANING_DURATION_SEC - leaning_time)
                cv2.rectangle(frame, (w // 2 - 120, 10), (w // 2 + 120, 80), COLOR_BG, -1)
                draw_text_bold(frame, f"เวลา: {rem:.1f} วิ", (w // 2 - 80, 25), 0.7,
                               COLOR_AMBER if rem > 10 else COLOR_RED)
                draw_text_bold(frame, f"หยิบได้: {ar_score} ครั้ง", (w // 2 - 80, 60), 0.6, COLOR_TEAL)
            elif mode == 2 and gait_state == 3:
                cv2.rectangle(frame, (w // 2 - 250, 10), (w // 2 + 250, 60), COLOR_BG, -1)
                draw_text_bold(frame,
                               f"เวลา: {gait_time:.1f} วิ | ก้าว: {step_count} | ระยะ: {walked_distance_m:.1f} ม.",
                               (w // 2 - 230, 25), 0.7, COLOR_GREEN)
            elif mode == 3 and m3_state == 2:
                rem = max(0.0, M3_DURATION_SEC - m3_time)
                cv2.rectangle(frame, (10, 10), (200, 80), COLOR_BG, -1)
                draw_text_bold(frame, f"เวลา: {rem:.1f} วิ", (30, 25), 0.7, COLOR_AMBER if rem > 10 else COLOR_RED)
                draw_text_bold(frame, f"คะแนน: {m3_hits}", (30, 60), 0.7, COLOR_TEAL)

        cv2.imshow("Balance Detection Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            mode = 1;
            leaning_state = 0;
            final_score = None;
            final_ar_score = None;
            current_instruction = None;
            is_cooldown = False
            ar_targets = []
            baseline_lean_lat = 0.0;
            lean_calibration_buf.clear()
            hip_buf.clear();
            sh_buf.clear()
        elif key == ord('2'):
            mode = 2;
            gait_state = 0;
            gait_time = 0.0;
            step_count = 0;
            step_count_steady = 0;
            last_step_time = 0.0
            current_tracking_part = "None";
            current_gait_phase = "รอทดสอบ"
            walked_distance_m = 0.0;
            gait_start_distance_m = 0.0;
            current_step_length = 0.0
            step_y_history.clear()
        elif key == ord('3'):
            mode = 3;
            m3_state = 0;
            current_instruction = None;
            is_cooldown = False
            baseline_lean_lat = 0.0;
            lean_calibration_buf.clear()
            hip_buf.clear();
            sh_buf.clear()
        elif key in (ord('g'), ord('G')):
            show_guide = not show_guide
        elif key == ord(' '):
            if mode == 1:
                if leaning_state == 0 or final_score is not None:
                    leaning_state = 1;
                    ready_start_t = None;
                    final_score = None;
                    final_ar_score = None;
                    is_cooldown = False
                    ar_targets = []
                    baseline_lean_lat = 0.0;
                    lean_calibration_buf.clear()

            elif mode == 2:
                if gait_state == 0 or gait_state == 4:
                    gait_state = 1
                    last_step_time = 0.0
                    step_y_history.clear()
                    walked_distance_m = 0.0;
                    current_step_length = 0.0
                elif gait_state == 3:
                    gait_state = 4

            elif mode == 3:
                if m3_state == 0 or m3_state == 3:
                    m3_state = 1;
                    m3_countdown_start = now;
                    m3_targets = [];
                    current_instruction = None;
                    is_cooldown = False
                    baseline_lean_lat = 0.0;
                    lean_calibration_buf.clear()
                    hip_buf.clear();
                    sh_buf.clear()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()