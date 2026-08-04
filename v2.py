import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

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

# โหมด 3 Constants
M3_DURATION_SEC = 60.0
M3_AR_TIMEOUT = 5.0
AR_COOLDOWN_SEC = 0.5

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
mode = 1
show_guide = True
current_instruction = None

# Mode 1
leaning_state = 0
ready_start_t = None
countdown_start_t = None
leaning_start = None
leaning_time = 0.0
ar_score = 0
ar_targets = []
final_score = None
final_ar_score = None

# Cooldown State
ar_cooldown_start = None
is_cooldown = False

# Mode 2
step_count = 0
prev_ankle_y = None
gait_start = None
gait_running = False
gait_time = 0.0

# Mode 3: 0=พร้อม, 1=นับถอยหลัง, 2=กำลังทดสอบ, 3=เสร็จสิ้น
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

MODE_NAMES = {1: "โน้มตัวซ้าย/ขวา + AR (60s)",
              2: "ก้าวเดิน (Gait)",
              3: "Dynamic Reach & Cognitive AR"}


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def lm_px(lm, w, h): return int(lm.x * w), int(lm.y * h)


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


def calc_balance_score(lat, rms_combined):
    p = min(100.0, (max(0.0, abs(lat) - LOS_LATERAL) / LOS_LATERAL) * 100) * LOS_SCORE_WEIGHT + \
        min(100.0, (rms_combined / RMS_MAX) * 100) * RMS_SCORE_WEIGHT
    return max(0, round(100 - p))


def score_color(
        s): return COLOR_GREEN if s >= 80 else COLOR_AMBER if s >= 60 else COLOR_ORANGE if s >= 40 else COLOR_RED


def score_label(s): return "ดี" if s >= 80 else "ปานกลาง" if s >= 60 else "เสี่ยง" if s >= 40 else "เสี่ยงสูง"


def detect_step(lms):
    global step_count, prev_ankle_y
    avg = (lms[mp_pose.PoseLandmark.LEFT_ANKLE].y + lms[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2
    if prev_ankle_y is not None and abs(avg - prev_ankle_y) > 0.015: step_count += 1
    prev_ankle_y = avg


def calc_trunk_sway_simple(lms):
    return abs(((lms[11].x + lms[12].x) / 2) - ((lms[23].x + lms[24].x) / 2)) * 100


def spawn_target(w, h, force_color=None, existing_targets=None):
    min_dist_sq = 0.20 ** 2
    for _ in range(50):
        tx = float(np.random.uniform(0.15, 0.85))
        ty = float(np.random.uniform(0.2, 0.8))
        too_close = False
        if existing_targets:
            for (ex_x, ex_y, _, _) in existing_targets:
                if (tx - ex_x) ** 2 + (ty - ex_y) ** 2 < min_dist_sq:
                    too_close = True
                    break
        if not too_close: break

    if force_color is None:
        cname, cbgr = AR_COLORS[np.random.randint(len(AR_COLORS))]
    else:
        cname, cbgr = force_color
    return (tx, ty, cname, cbgr)


def reset_ar_task(w, h):
    inst_color = AR_COLORS[np.random.randint(len(AR_COLORS))]
    targets = []
    targets.append(spawn_target(w, h, force_color=inst_color, existing_targets=targets))
    distractor_colors = [c for c in AR_COLORS if c[0] != inst_color[0]]
    for _ in range(2):
        dist_color = distractor_colors[np.random.randint(len(distractor_colors))]
        targets.append(spawn_target(w, h, force_color=dist_color, existing_targets=targets))
    np.random.shuffle(targets)
    return inst_color, targets


def check_reach(lms, targets, w, h):
    lw, rw = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    hits = []
    for i, (tx, ty, cname, cbgr) in enumerate(targets):
        if any(np.hypot(wr.x - tx, wr.y - ty) < 0.06 for wr in [lw, rw]): hits.append(i)
    return hits


def draw_reach_targets(frame, targets, w, h):
    for (tx, ty, cname, cbgr) in targets:
        px, py = int(tx * w), int(ty * h)
        cv2.circle(frame, (px, py), 26, cbgr, -1)
        cv2.circle(frame, (px, py), 26, COLOR_WHITE, 2)


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
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


def draw_bar(frame, x, y, value, max_val, bw, bh, color):
    fill = int(np.clip(value / max_val * bw, 0, bw))
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_GRAY, -1)
    if fill > 0: cv2.rectangle(frame, (x, y), (x + fill, y + bh), color, -1)
    cv2.line(frame, (x, y - 2), (x, y + bh + 2), COLOR_WHITE, 2)


def draw_score_ring(frame, cx, cy, score, r=28):
    color = score_color(score)
    cv2.circle(frame, (cx, cy), r, COLOR_GRAY, 3)
    for a in range(0, int(360 * score / 100), 3):
        rad = np.radians(a - 90)
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


def draw_info_panel(frame, mode, lean_lat, balance_score, hip_rms, step_count, gait_time, simple_sway, w, h):
    pw = 290
    if mode == 1:
        ph = 300
    elif mode == 2:
        ph = 195
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
            draw_text_th(frame, "กรุณายืนนิ่งๆ รอสัญญาณ...", (20, y), color=COLOR_AMBER)
        elif leaning_state == 2:
            rem_cd = max(1, int(np.ceil(3.0 - (time.time() - countdown_start_t)))) if countdown_start_t else 1
            draw_text_bold(frame, f"เตรียมตัว! {rem_cd}", (20, y), 0.6, COLOR_ORANGE)
        elif leaning_state == 3:
            rem = max(0.0, LEANING_DURATION_SEC - leaning_time)
            draw_text_th(frame, f"ทดสอบ: {rem:.1f} วินาที", (20, y), color=COLOR_AMBER if rem > 10 else COLOR_RED)
        elif final_score is not None:
            draw_text_th(frame, "เสร็จสิ้น", (20, y), color=COLOR_GREEN)
        else:
            draw_text_th(frame, "กด [SPACE] เริ่ม 60s", (20, y), color=COLOR_GRAY)

        display_score = final_score if final_score is not None else balance_score
        display_ar = final_ar_score if final_ar_score is not None else ar_score

        y += 24;
        draw_text_th(frame, f"AR หยิบถูก: {display_ar} ครั้ง", (20, y), color=COLOR_TEAL)
        y += 26;
        BW = 245;
        abs_lat = abs(lean_lat)
        col_lat = COLOR_RED if abs_lat > LOS_LATERAL else COLOR_GREEN
        draw_text_th(frame, f"ซ้าย/ขวา: {lean_lat:+.1f}° limit ±{LOS_LATERAL}°", (20, y), color=col_lat)
        y += 19;
        draw_bar(frame, 20, y, abs_lat, LOS_LATERAL * 1.5, BW, 9, col_lat)
        y += 24;
        rms_col = COLOR_GREEN if hip_rms < RMS_MAX * 0.5 else COLOR_AMBER if hip_rms < RMS_MAX else COLOR_RED
        draw_text_th(frame, f"RMS Hip:{hip_rms:.4f}", (20, y), 0.44, rms_col)
        y += 18;
        draw_bar(frame, 20, y, hip_rms, RMS_MAX, BW, 9, rms_col)
        y += 24;
        sc = score_color(display_score)
        draw_score_ring(frame, 48, y + 25, display_score)
        draw_text_bold(frame, f"คะแนน: {display_score}/100", (90, y + 16), 0.55, sc)
        draw_text_th(frame, f"ระดับ: {score_label(display_score)}", (90, y + 40), color=sc)

    elif mode == 2:
        timer_str = f"{gait_time:.1f} วินาที" if gait_running or gait_time > 0 else "กด [SPACE] เริ่ม"
        draw_text_th(frame, f"เวลา: {timer_str}", (20, y));
        y += 26
        draw_text_th(frame, f"ก้าว: {step_count} ก้าว", (20, y), color=COLOR_AMBER);
        y += 26
        draw_text_th(frame, f"Trunk sway: {simple_sway:.1f}%", (20, y), color=COLOR_TEAL);
        y += 26
        status = "กำลังทดสอบ..." if gait_running else "เสร็จสิ้น" if gait_time > 0 else "พร้อมทดสอบ"
        draw_text_th(frame, f"สถานะ: {status}", (20, y))

    elif mode == 3:
        if m3_state == 1:
            rem_cd = max(1, int(np.ceil(3.0 - (time.time() - m3_countdown_start)))) if m3_countdown_start else 1
            draw_text_bold(frame, f"เตรียมตัว! {rem_cd}", (20, y), 0.6, COLOR_ORANGE)
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
    items = [("[1] โน้มตัว+สี", mode == 1), ("[2] ก้าวเดิน", mode == 2),
             ("[3] เอื้อมมือ+สี", mode == 3), ("[G] Guide", show_guide),
             ("[SPACE] Action", False), ("[Q] ออก", False)]
    px = w - 195
    draw_panel(frame, px - 5, 10, 190, len(items) * 26 + 16, COLOR_BG, 0.80)
    for i, (lbl, active) in enumerate(items):
        draw_text_th(frame, lbl, (px, 36 + i * 26), color=COLOR_TEAL if active else COLOR_GRAY)


def main():
    global mode, show_guide, step_count, prev_ankle_y
    global gait_start, gait_running, gait_time, current_instruction
    global leaning_state, ready_start_t, countdown_start_t, leaning_start, leaning_time, ar_score, ar_targets, final_score, final_ar_score
    global is_cooldown, ar_cooldown_start
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

            lean_lat = calc_lean_lat(lms)
            update_sway_buffers(lms, now)
            hip_rms, sh_rms, rms_comb = calc_rms_sway()
            current_score = calc_balance_score(lean_lat, rms_comb)

            if is_cooldown and now - ar_cooldown_start >= AR_COOLDOWN_SEC:
                is_cooldown = False

            if mode == 1:
                if leaning_state == 1:
                    if hip_rms < RMS_MAX * 0.8:
                        if ready_start_t is None:
                            ready_start_t = now
                        elif now - ready_start_t >= 1.0:
                            leaning_state = 2;
                            countdown_start_t = now
                    else:
                        ready_start_t = None
                    balance_score = current_score

                elif leaning_state == 2:
                    if countdown_start_t is not None and now - countdown_start_t >= 3.0:
                        leaning_state = 3;
                        leaning_start = now;
                        leaning_time = 0.0;
                        ar_score = 0
                        current_instruction, ar_targets = reset_ar_task(w, h)
                        hip_buf.clear();
                        sh_buf.clear()
                    balance_score = current_score

                elif leaning_state == 3:
                    leaning_time = now - leaning_start
                    balance_score = current_score

                    if not is_cooldown:
                        hits = check_reach(lms, ar_targets, w, h)
                        if hits:
                            hit_idx = hits[0]
                            if ar_targets[hit_idx][2] == current_instruction[0]:
                                ar_score += 1
                                ar_targets = []
                                is_cooldown = True
                                ar_cooldown_start = now
                                current_instruction, ar_targets = reset_ar_task(w, h)
                            else:
                                ar_targets.pop(hit_idx)
                                dist_color = [c for c in AR_COLORS if c[0] != current_instruction[0]][
                                    np.random.randint(4)]
                                ar_targets.append(
                                    spawn_target(w, h, force_color=dist_color, existing_targets=ar_targets))

                    if not is_cooldown:
                        draw_reach_targets(frame, ar_targets, w, h)

                    for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        px, py = lm_px(lms[lm_id], w, h);
                        cv2.circle(frame, (px, py), 12, COLOR_CORAL, -1)

                    if leaning_time >= LEANING_DURATION_SEC:
                        leaning_state = 0;
                        final_score = balance_score;
                        final_ar_score = ar_score
                        ar_targets = [];
                        current_instruction = None;
                        is_cooldown = False
                else:
                    balance_score = current_score

            elif mode == 2:
                detect_step(lms)
                simple_sway = calc_trunk_sway_simple(lms)
                if gait_running: gait_time = now - gait_start
                for lm_id in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]:
                    px, py = lm_px(lms[lm_id], w, h);
                    cv2.circle(frame, (px, py), 14, COLOR_AMBER, -1)

            elif mode == 3:
                balance_score = current_score
                if m3_state == 1:
                    # สถานะนับถอยหลัง 3 วินาที
                    if now - m3_countdown_start >= 3.0:
                        m3_state = 2
                        m3_start_t = now
                        m3_time = 0.0
                        m3_hits = 0
                        m3_misses = 0
                        m3_rx_times = []
                        m3_bal_scores = []
                        is_cooldown = False
                        current_instruction, m3_targets = reset_ar_task(w, h)
                        m3_spawn_t = now
                        hip_buf.clear();
                        sh_buf.clear()

                elif m3_state == 2:
                    # สถานะกำลังทดสอบ 60 วินาที
                    m3_time = now - m3_start_t
                    m3_bal_scores.append(current_score)

                    if not is_cooldown:
                        hits = check_reach(lms, m3_targets, w, h)
                        if hits:
                            hit_idx = hits[0]
                            if m3_targets[hit_idx][2] == current_instruction[0]:
                                m3_hits += 1
                                m3_rx_times.append(now - m3_spawn_t)
                                m3_targets = []
                                is_cooldown = True
                                ar_cooldown_start = now
                                current_instruction, m3_targets = reset_ar_task(w, h)
                                m3_spawn_t = now + AR_COOLDOWN_SEC
                            else:
                                m3_misses += 1
                                m3_targets.pop(hit_idx)
                                dist_color = [c for c in AR_COLORS if c[0] != current_instruction[0]][
                                    np.random.randint(4)]
                                m3_targets.append(
                                    spawn_target(w, h, force_color=dist_color, existing_targets=m3_targets))

                        elif now - m3_spawn_t > M3_AR_TIMEOUT:
                            m3_misses += 1
                            m3_targets = []
                            is_cooldown = True
                            ar_cooldown_start = now
                            current_instruction, m3_targets = reset_ar_task(w, h)
                            m3_spawn_t = now + AR_COOLDOWN_SEC

                    if not is_cooldown:
                        draw_reach_targets(frame, m3_targets, w, h)

                    for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
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

        # วาดคำสั่งใหญ่ๆ กลางจอ
        if (mode == 1 and leaning_state == 3) or (mode == 3 and m3_state == 2):
            if current_instruction:
                if is_cooldown:
                    cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
                    draw_text_bold(frame, "ดึงมือกลับ...", (w // 2 - 60, 30), 0.8, COLOR_GRAY)
                else:
                    cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
                    draw_text_bold(frame, f"คำสั่ง: แตะเป้าหมาย {current_instruction[0]}", (w // 2 - 170, 30), 0.8,
                                   current_instruction[1])
                    if mode == 3:
                        rem_obj = max(0, M3_AR_TIMEOUT - (now - m3_spawn_t))
                        cv2.rectangle(frame, (w // 2 - 170, 70),
                                      (w // 2 - 170 + int(340 * (rem_obj / M3_AR_TIMEOUT)), 75), current_instruction[1],
                                      -1)

        # โชว์เวลานับถอยหลังตัวใหญ่ๆ กลางจอ
        if mode == 3 and m3_state == 1:
            rem_cd = max(1, int(np.ceil(3.0 - (now - m3_countdown_start))))
            cv2.rectangle(frame, (w // 2 - 200, 10), (w // 2 + 200, 80), COLOR_BG, -1)
            draw_text_bold(frame, f"เตรียมตัวเริ่มใน... {rem_cd}", (w // 2 - 120, 30), 0.8, COLOR_ORANGE)

        draw_info_panel(frame, mode, lean_lat, balance_score, hip_rms, step_count, gait_time, simple_sway, w, h)
        draw_controls(frame, w)
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
            hip_buf.clear();
            sh_buf.clear()
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3;
            m3_state = 0;
            current_instruction = None;
            is_cooldown = False
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
            elif mode == 2:
                if not gait_running:
                    gait_start = now;
                    gait_running = True;
                    step_count = 0;
                    prev_ankle_y = None;
                    gait_time = 0.0
                else:
                    gait_running = False
            elif mode == 3:
                # เริ่มจับเวลานับถอยหลังเมื่อกด Spacebar ในโหมด 3
                if m3_state == 0 or m3_state == 3:
                    m3_state = 1
                    m3_countdown_start = now
                    m3_targets = []
                    current_instruction = None
                    is_cooldown = False
                    hip_buf.clear();
                    sh_buf.clear()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()