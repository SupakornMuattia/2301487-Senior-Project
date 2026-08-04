"""
Balance Detection Demo — MediaPipe BlazePose
============================================
โปรเจก: แอปตรวจจับและวัดการทรงตัวสำหรับผู้สูงอายุ
วิชา: 2301399 Project Proposal กลุ่มที่ 109

วิธีรัน:
    pip install mediapipe opencv-python numpy
    python balance_demo.py

ควบคุม:
    [1] โหมดโน้มตัว (Leaning)
    [2] โหมดก้าวเดิน (Gait)
    [3] โหมดเอื้อมมือ (Reach)
    [G] เปิด/ปิด Guide line
    [Q] ออกจากโปรแกรม
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
# ─────────────────────────────────────────────
# MediaPipe Setup
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

# ─────────────────────────────────────────────
# สีและการตั้งค่า UI
# ─────────────────────────────────────────────
COLOR_BG       = (15, 15, 25)
COLOR_TEAL     = (0, 200, 170)
COLOR_CORAL    = (60, 100, 230)
COLOR_AMBER    = (30, 180, 230)
COLOR_WHITE    = (240, 240, 240)
COLOR_GRAY     = (100, 100, 120)
COLOR_GREEN    = (50, 200, 100)
COLOR_RED      = (60, 60, 220)
COLOR_GUIDE    = (200, 200, 50)

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD      = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
mode           = 1          # 1=leaning, 2=gait, 3=reach
show_guide     = True
step_count     = 0
prev_ankle_y   = None
gait_start     = None
gait_running   = False
gait_time      = 0.0
reach_targets  = []         # list of (x_norm, y_norm)
reach_hit      = []
reach_spawn_t  = 0.0

MODE_NAMES = {
    1: "โน้มตัว (Leaning)",
    2: "ก้าวเดิน (Gait)",
    3: "เอื้อมมือ (Reach & AR)"
}

# ─────────────────────────────────────────────
# Helper: landmark → pixel
# ─────────────────────────────────────────────
def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# ─────────────────────────────────────────────
# คำนวณมุมระหว่าง 3 จุด
# ─────────────────────────────────────────────
def calc_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# ─────────────────────────────────────────────
# คำนวณมุมเอียงซ้าย/ขวา (lateral lean)
# จาก shoulder midpoint → hip midpoint
# ─────────────────────────────────────────────
def calc_lean_angle(lms):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip_mid      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])

    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]   # y เพิ่มลงล่าง
    angle = np.degrees(np.arctan2(dx, dy + 1e-6))
    return angle   # บวก = เอียงขวา, ลบ = เอียงซ้าย

# ─────────────────────────────────────────────
# ตรวจจับก้าว (Gait)
# ─────────────────────────────────────────────
def detect_step(lms):
    global step_count, prev_ankle_y
    la = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]
    avg_y = (la.y + ra.y) / 2
    if prev_ankle_y is not None:
        diff = abs(avg_y - prev_ankle_y)
        if diff > 0.015:
            step_count += 1
    prev_ankle_y = avg_y

# ─────────────────────────────────────────────
# วัด trunk sway (การแกว่งลำตัว)
# ─────────────────────────────────────────────
def calc_trunk_sway(lms):
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_cx = (lh.x + rh.x) / 2
    sh_cx  = (ls.x + rs.x) / 2
    return abs(sh_cx - hip_cx) * 100  # % of frame width

# ─────────────────────────────────────────────
# ตรวจจับมือเอื้อมถึง target
# ─────────────────────────────────────────────
def check_reach(lms, targets, w, h):
    hit_indices = []
    lw = lms[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    for i, (tx, ty) in enumerate(targets):
        for wrist in [lw, rw]:
            dist = np.hypot(wrist.x - tx, wrist.y - ty)
            if dist < 0.08:
                hit_indices.append(i)
    return hit_indices

# ─────────────────────────────────────────────
# วาด UI panels
# ─────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, color, alpha=0.35):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


def put_thai_text(frame, text, pos, font_size=20, color=(240, 240, 240)):
    # 1. แปลงสีจาก BGR (OpenCV) เป็น RGB (Pillow)
    b, g, r = color

    # 2. แปลง Frame เป็น PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 3. โหลดฟอนต์ (เปลี่ยนเส้นทางฟอนต์ตาม OS ของคุณ)
    # Windows มักจะมีฟอนต์ Tahoma อยู่แล้ว: "tahoma.ttf" หรือ "updil.ttf"
    # Mac มักจะมี "Thonburi.ttf" หรือ "Ayuthaya.ttf"
    try:
        font = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        # ถ้าหาฟอนต์ไม่เจอ ให้ fallback ไปใช้ฟอนต์ default (อาจจะแสดงไทยไม่ได้)
        font = ImageFont.load_default()
        print("คำเตือน: ไม่พบไฟล์ฟอนต์ภาษาไทย กรุณาใส่ไฟล์ .ttf ไว้ในโฟลเดอร์เดียวกับโค้ด")

    # 4. วาดตัวอักษร
    draw.text(pos, text, font=font, fill=(r, g, b))

    # 5. แปลงกลับเป็น OpenCV Image และทับลงไปใน frame เดิม
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:] = frame_bgr[:]  # อัปเดตข้อมูลทับ frame ตัวแปรเดิม

def draw_text_th(frame, text, pos, scale=0.55, color=COLOR_WHITE, thickness=1):
    # ปรับ scale จาก OpenCV (0.55) เป็น font_size ของ Pillow (ประมาณ 20-24)
    font_size = int(scale * 40)
    put_thai_text(frame, text, pos, font_size=font_size, color=color)

def draw_text_bold(frame, text, pos, scale=0.7, color=COLOR_WHITE, thickness=2):
    # ปรับให้ใหญ่ขึ้นนิดหน่อยแทนตัวหนา
    font_size = int(scale * 45)
    put_thai_text(frame, text, pos, font_size=font_size, color=color)

# ─────────────────────────────────────────────
# วาด guide line สำหรับแต่ละโหมด
# ─────────────────────────────────────────────
def draw_guide(frame, mode, w, h):
    if mode == 1:
        # เส้นแนวตั้งกึ่งกลาง (center axis)
        cx = w // 2
        cv2.line(frame, (cx, 50), (cx, h - 50), COLOR_GUIDE, 1, cv2.LINE_AA)
        # เส้นบอกองศาซ้าย/ขวา ±10°
        for deg, label in [(-10, "-10"), (10, "+10")]:
            offset = int(np.tan(np.radians(deg)) * (h // 2))
            cv2.line(frame, (cx, h // 2), (cx + offset, h - 50),
                     COLOR_GUIDE, 1, cv2.LINE_AA)
        draw_text_th(frame, "Guide: ยืนตรงกลาง", (cx - 70, 40),
                     color=COLOR_GUIDE)

    elif mode == 2:
        # เส้นทางเดิน
        cx = w // 2
        for offset in [-40, 40]:
            cv2.line(frame, (cx + offset, 80), (cx + offset, h - 80),
                     COLOR_GUIDE, 1, cv2.LINE_AA)
        draw_text_th(frame, "Guide: เดินตามเส้น 5 เมตร", (cx - 100, 40),
                     color=COLOR_GUIDE)

    elif mode == 3:
        draw_text_th(frame, "Guide: เอื้อมมือแตะวงกลม", (20, 40),
                     color=COLOR_GUIDE)

# ─────────────────────────────────────────────
# วาด reach targets (วัตถุ AR จำลอง)
# ─────────────────────────────────────────────
def spawn_target(w, h):
    zones = [(0.25, 0.3), (0.75, 0.3), (0.5, 0.5),
             (0.2, 0.6),  (0.8, 0.6)]
    tx, ty = zones[np.random.randint(len(zones))]
    tx += np.random.uniform(-0.05, 0.05)
    ty += np.random.uniform(-0.05, 0.05)
    return (float(np.clip(tx, 0.1, 0.9)),
            float(np.clip(ty, 0.15, 0.85)))

def draw_reach_targets(frame, targets, hits, w, h):
    for i, (tx, ty) in enumerate(targets):
        px, py = int(tx * w), int(ty * h)
        color = COLOR_GREEN if i in hits else COLOR_CORAL
        cv2.circle(frame, (px, py), 28, color, -1)
        cv2.circle(frame, (px, py), 28, COLOR_WHITE, 2)
        draw_text_bold(frame, "AR", (px - 13, py + 7), 0.5, COLOR_WHITE)

# ─────────────────────────────────────────────
# วาด info panel ซ้ายบน
# ─────────────────────────────────────────────
def draw_info_panel(frame, mode, lean_angle, step_count,
                    gait_time, sway, reach_hit, w, h):
    pw, ph = 260, 180
    draw_panel(frame, 10, 10, pw, ph, COLOR_BG, 0.75)

    # ชื่อโหมด
    draw_text_bold(frame, f"Mode: {mode}", (20, 38), 0.55, COLOR_TEAL)
    draw_text_th(frame, MODE_NAMES[mode], (20, 62), 0.5, COLOR_WHITE)

    y = 90
    if mode == 1:
        direction = "ขวา" if lean_angle > 0 else "ซ้าย"
        abs_ang = abs(lean_angle)
        color = COLOR_RED if abs_ang > 15 else COLOR_GREEN
        draw_text_th(frame, f"มุมเอียง: {lean_angle:+.1f} deg ({direction})",
                     (20, y), color=color); y += 26
        # แถบแสดงมุม
        bar_w = int(np.clip(abs_ang / 30 * 100, 0, 100))
        cv2.rectangle(frame, (20, y), (120, y + 12), COLOR_GRAY, -1)
        cv2.rectangle(frame, (20, y), (20 + bar_w, y + 12), color, -1)
        y += 30
        status = "เกินเกณฑ์!" if abs_ang > 15 else "ปกติ"
        draw_text_th(frame, f"สถานะ: {status}", (20, y), color=color); y += 26
        draw_text_th(frame, "เกณฑ์: > 8 deg = เสี่ยง", (20, y),
                     color=COLOR_GRAY)

    elif mode == 2:
        timer_str = f"{gait_time:.1f} วินาที" if gait_running or gait_time > 0 \
                    else "กด [SPACE] เริ่ม"
        draw_text_th(frame, f"เวลา: {timer_str}", (20, y)); y += 26
        draw_text_th(frame, f"ก้าว: {step_count} ก้าว",
                     (20, y), color=COLOR_AMBER); y += 26
        draw_text_th(frame, f"Trunk sway: {sway:.1f}%",
                     (20, y), color=COLOR_TEAL); y += 26
        status = "กำลังทดสอบ..." if gait_running else \
                 ("เสร็จสิ้น" if gait_time > 0 else "พร้อมทดสอบ")
        draw_text_th(frame, f"สถานะ: {status}", (20, y))

    elif mode == 3:
        hits = len(reach_hit)
        total = len(reach_targets)
        draw_text_th(frame, f"แตะได้: {hits}/{total} เป้าหมาย",
                     (20, y), color=COLOR_GREEN); y += 26
        draw_text_th(frame, "กด [SPACE] สร้างเป้าใหม่",
                     (20, y), color=COLOR_GRAY)

# ─────────────────────────────────────────────
# วาด control hints ขวาบน
# ─────────────────────────────────────────────
def draw_controls(frame, w):
    controls = [
        ("[1] โน้มตัว", mode == 1),
        ("[2] ก้าวเดิน", mode == 2),
        ("[3] เอื้อมมือ", mode == 3),
        ("[G] Guide", show_guide),
        ("[SPACE] Action", False),
        ("[Q] ออก", False),
    ]
    px = w - 190
    draw_panel(frame, px - 5, 10, 185, len(controls) * 26 + 16, COLOR_BG, 0.75)
    for i, (label, active) in enumerate(controls):
        color = COLOR_TEAL if active else COLOR_GRAY
        draw_text_th(frame, label, (px, 36 + i * 26), color=color)

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    global mode, show_guide, step_count, prev_ankle_y
    global gait_start, gait_running, gait_time
    global reach_targets, reach_hit, reach_spawn_t

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่พบกล้อง — กรุณาเสียบกล้องแล้วรันใหม่")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print(" Balance Detection Demo — MediaPipe BlazePose")
    print("=" * 50)
    print(" [1] โหมดโน้มตัว   [2] โหมดก้าวเดิน   [3] โหมดเอื้อมมือ")
    print(" [G] Guide line     [SPACE] Action       [Q] ออก")
    print("=" * 50)

    lean_angle = 0.0
    sway = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ─── Pose detection ───
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        # ─── วาด guide ───
        if show_guide:
            draw_guide(frame, mode, w, h)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            # วาด skeleton
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=COLOR_TEAL, thickness=4, circle_radius=4),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=COLOR_WHITE, thickness=2)
            )

            # ─── โหมด 1: Leaning ───
            if mode == 1:
                lean_angle = calc_lean_angle(lms)
                # วาดเส้นแกนลำตัว
                ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
                rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
                sh_mid = (int((ls.x + rs.x) / 2 * w),
                          int((ls.y + rs.y) / 2 * h))
                hp_mid = (int((lh.x + rh.x) / 2 * w),
                          int((lh.y + rh.y) / 2 * h))
                color = COLOR_RED if abs(lean_angle) > 8 else COLOR_GREEN
                cv2.line(frame, sh_mid, hp_mid, color, 3, cv2.LINE_AA)
                cv2.circle(frame, sh_mid, 8, color, -1)
                cv2.circle(frame, hp_mid, 8, color, -1)

            # ─── โหมด 2: Gait ───
            elif mode == 2:
                detect_step(lms)
                sway = calc_trunk_sway(lms)
                if gait_running:
                    gait_time = time.time() - gait_start

                # ไฮไลต์ข้อเท้า
                for lm_id in [mp_pose.PoseLandmark.LEFT_ANKLE,
                               mp_pose.PoseLandmark.RIGHT_ANKLE]:
                    px, py = lm_px(lms[lm_id], w, h)
                    cv2.circle(frame, (px, py), 14, COLOR_AMBER, -1)
                    cv2.circle(frame, (px, py), 14, COLOR_WHITE, 2)

            # ─── โหมด 3: Reach ───
            elif mode == 3:
                # spawn target ทุก 4 วินาที ถ้ายังไม่มี
                if not reach_targets or \
                   time.time() - reach_spawn_t > 4.0:
                    reach_targets = [spawn_target(w, h)
                                     for _ in range(3)]
                    reach_hit = []
                    reach_spawn_t = time.time()

                hit_idx = check_reach(lms, reach_targets, w, h)
                for i in hit_idx:
                    if i not in reach_hit:
                        reach_hit.append(i)

                draw_reach_targets(frame, reach_targets,
                                   reach_hit, w, h)

                # ไฮไลต์ข้อมือ
                for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST,
                               mp_pose.PoseLandmark.RIGHT_WRIST]:
                    px, py = lm_px(lms[lm_id], w, h)
                    cv2.circle(frame, (px, py), 12, COLOR_CORAL, -1)
                    cv2.circle(frame, (px, py), 12, COLOR_WHITE, 2)

        else:
            # ไม่พบร่างกาย
            draw_text_bold(frame, "ไม่พบผู้ใช้ — กรุณายืนหน้ากล้อง",
                           (w // 2 - 200, h // 2),
                           color=COLOR_RED)

        # ─── วาด UI ───
        draw_info_panel(frame, mode, lean_angle, step_count,
                        gait_time, sway, reach_hit, w, h)
        draw_controls(frame, w)

        # ─── FPS ───
        cv2.putText(frame, f"MediaPipe BlazePose | Balance Demo",
                    (w // 2 - 160, h - 15), FONT, 0.45,
                    COLOR_GRAY, 1, cv2.LINE_AA)

        cv2.imshow("Balance Detection Demo", frame)

        # ─── Keyboard ───
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            mode = 1; lean_angle = 0.0
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3; reach_targets = []; reach_hit = []
        elif key == ord('g') or key == ord('G'):
            show_guide = not show_guide
        elif key == ord(' '):
            if mode == 2:
                if not gait_running:
                    gait_start = time.time()
                    gait_running = True
                    step_count = 0
                    prev_ankle_y = None
                    gait_time = 0.0
                    print(f"  เริ่มจับเวลา Gait test...")
                else:
                    gait_running = False
                    print(f"  หยุด — เวลา: {gait_time:.1f}s "
                          f"| ก้าว: {step_count}")
            elif mode == 3:
                reach_targets = [spawn_target(w, h) for _ in range(3)]
                reach_hit = []
                reach_spawn_t = time.time()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nปิดโปรแกรมเรียบร้อย")

if __name__ == "__main__":
    main()