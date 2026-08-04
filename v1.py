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
# สี UI
# ─────────────────────────────────────────────
COLOR_BG     = (15, 15, 25)
COLOR_TEAL   = (0, 200, 170)
COLOR_CORAL  = (60, 100, 230)
COLOR_AMBER  = (30, 180, 230)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (100, 100, 120)
COLOR_GREEN  = (50, 200, 100)
COLOR_RED    = (60, 60, 220)
COLOR_GUIDE  = (200, 200, 50)
COLOR_ORANGE = (30, 140, 255)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
# ค่าคงที่
# ─────────────────────────────────────────────
LOS_LATERAL   = 8.0

RMS_WINDOW_SEC   = 2.0
RMS_MAX          = 0.12
RMS_HIP_WEIGHT   = 0.6
RMS_SH_WEIGHT    = 0.4

LOS_SCORE_WEIGHT = 0.8
RMS_SCORE_WEIGHT = 0.2

LEANING_DURATION_SEC = 60.0

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
mode          = 1
show_guide    = True

# Mode 1: Leaning + AR
leaning_state     = 0     # 0: พร้อม, 1: รอความนิ่ง, 2: นับถอยหลัง 3 วิ, 3: กำลังทดสอบ
ready_start_t     = None
countdown_start_t = None
leaning_start     = None
leaning_time      = 0.0
ar_score          = 0
ar_targets        = []
final_score       = None
final_ar_score    = None

# Mode 2 & 3
step_count    = 0
prev_ankle_y  = None
gait_start    = None
gait_running  = False
gait_time     = 0.0
reach_targets = []
reach_hit     = []
reach_spawn_t = 0.0

hip_buf = deque()
sh_buf  = deque()

MODE_NAMES = {1: "โน้มตัวซ้าย/ขวา + หยิบของ (60s)",
              2: "ก้าวเดิน (Gait)",
              3: "เอื้อมมือ (Reach & AR)"}

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def calc_lean_lat(lms):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    sh = np.array([(ls.x+rs.x)/2, (ls.y+rs.y)/2, (ls.z+rs.z)/2])
    hp = np.array([(lh.x+rh.x)/2, (lh.y+rh.y)/2, (lh.z+rh.z)/2])
    d  = sh - hp
    dy = abs(d[1]) + 1e-6
    return float(np.degrees(np.arctan2(d[0], dy)))

def update_sway_buffers(lms, now):
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_buf.append(((lh.x+rh.x)/2, (lh.y+rh.y)/2, now))
    sh_buf.append( ((ls.x+rs.x)/2, (ls.y+rs.y)/2, now))
    cutoff = now - RMS_WINDOW_SEC
    while hip_buf and hip_buf[0][2] < cutoff: hip_buf.popleft()
    while sh_buf  and sh_buf[0][2]  < cutoff: sh_buf.popleft()

def calc_rms_sway():
    if len(hip_buf) < 5:
        return 0.0, 0.0, 0.0
    ha = np.array([(x, y) for x, y, _ in hip_buf])
    sa = np.array([(x, y) for x, y, _ in sh_buf])
    rms_h = float(np.sqrt(np.mean((ha[:,0]-ha[:,0].mean())**2 +
                                   (ha[:,1]-ha[:,1].mean())**2)))
    rms_s = float(np.sqrt(np.mean((sa[:,0]-sa[:,0].mean())**2 +
                                   (sa[:,1]-sa[:,1].mean())**2)))
    return rms_h, rms_s, rms_h * RMS_HIP_WEIGHT + rms_s * RMS_SH_WEIGHT

def calc_los_penalty(lat):
    over_lat  = max(0.0, abs(lat) - LOS_LATERAL)
    p = (over_lat / LOS_LATERAL) * 100
    return min(100.0, p)

def calc_rms_penalty(rms_combined):
    return min(100.0, (rms_combined / RMS_MAX) * 100)

def calc_balance_score(lat, rms_combined):
    p = calc_los_penalty(lat) * LOS_SCORE_WEIGHT + \
        calc_rms_penalty(rms_combined) * RMS_SCORE_WEIGHT
    return max(0, round(100 - p))

def score_color(s):
    return COLOR_GREEN if s >= 80 else COLOR_AMBER if s >= 60 \
           else COLOR_ORANGE if s >= 40 else COLOR_RED

def score_label(s):
    return "ดี" if s >= 80 else "ปานกลาง" if s >= 60 \
           else "เสี่ยง" if s >= 40 else "เสี่ยงสูง"

def detect_step(lms):
    global step_count, prev_ankle_y
    la  = lms[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra  = lms[mp_pose.PoseLandmark.RIGHT_ANKLE]
    avg = (la.y + ra.y) / 2
    if prev_ankle_y is not None and abs(avg - prev_ankle_y) > 0.015:
        step_count += 1
    prev_ankle_y = avg

def calc_trunk_sway_simple(lms):
    lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    return abs(((ls.x+rs.x)/2) - ((lh.x+rh.x)/2)) * 100

def spawn_target(w, h):
    zones = [(0.15,0.4),(0.85,0.4),(0.5,0.5),(0.2,0.6),(0.8,0.6)]
    tx, ty = zones[np.random.randint(len(zones))]
    return (float(np.clip(tx + np.random.uniform(-0.05,0.05), 0.1, 0.9)),
            float(np.clip(ty + np.random.uniform(-0.05,0.05), 0.15, 0.85)))

def check_reach(lms, targets, w, h):
    lw = lms[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    return [i for i,(tx,ty) in enumerate(targets)
            if any(np.hypot(wr.x-tx, wr.y-ty) < 0.08 for wr in [lw,rw])]

def draw_reach_targets(frame, targets, hits, w, h):
    for i,(tx,ty) in enumerate(targets):
        px,py = int(tx*w), int(ty*h)
        color = COLOR_GREEN if i in hits else COLOR_CORAL
        cv2.circle(frame,(px,py),28,color,-1)
        cv2.circle(frame,(px,py),28,COLOR_WHITE,2)
        draw_text_bold(frame,"AR",(px-13,py+7),0.5,COLOR_WHITE)

def put_thai_text(frame, text, pos, font_size=20, color=(240,240,240)):
    b, g, r = color
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    drw = ImageDraw.Draw(pil)
    try:
        fnt = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        fnt = ImageFont.load_default()
    drw.text(pos, text, font=fnt, fill=(r,g,b))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)[:]

def draw_text_th(frame, text, pos, scale=0.55, color=COLOR_WHITE, **_):
    put_thai_text(frame, text, pos, int(scale*40), color)

def draw_text_bold(frame, text, pos, scale=0.7, color=COLOR_WHITE, **_):
    put_thai_text(frame, text, pos, int(scale*45), color)

def draw_panel(frame, x, y, w, h, color, alpha=0.35):
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),color,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),color,1)

def draw_bar(frame, x, y, value, max_val, bw, bh, color):
    fill = int(np.clip(value/max_val*bw, 0, bw))
    cv2.rectangle(frame,(x,y),(x+bw,y+bh),COLOR_GRAY,-1)
    if fill > 0:
        cv2.rectangle(frame,(x,y),(x+fill,y+bh),color,-1)

def draw_limit_marker(frame, x, y, bw, bh):
    cv2.line(frame,(x,y-2),(x,y+bh+2),COLOR_WHITE,2)

def draw_score_ring(frame, cx, cy, score, r=28):
    color = score_color(score)
    cv2.circle(frame,(cx,cy),r,COLOR_GRAY,3)
    for a in range(0, int(360*score/100), 3):
        rad = np.radians(a-90)
        cv2.circle(frame,
                   (int(cx+r*np.cos(rad)), int(cy+r*np.sin(rad))),
                   2, color, -1)
    ox = 14 if score>=100 else 10 if score>=10 else 6
    cv2.putText(frame,str(score),(cx-ox,cy+5),FONT_BOLD,0.55,color,1,cv2.LINE_AA)

def draw_guide(frame, mode, w, h):
    if mode == 1:
        cx = w//2
        cv2.line(frame,(cx,50),(cx,h-50),COLOR_GUIDE,1,cv2.LINE_AA)
        for deg in [-LOS_LATERAL, LOS_LATERAL]:
            off = int(np.tan(np.radians(deg))*(h//2))
            cv2.line(frame,(cx,h//2),(cx+off,h-50),COLOR_GUIDE,1,cv2.LINE_AA)
        draw_text_th(frame,f"LoS ±{LOS_LATERAL}° lateral", (cx-100,40), color=COLOR_GUIDE)
    elif mode == 2:
        cx = w//2
        for off in [-40,40]:
            cv2.line(frame,(cx+off,80),(cx+off,h-80),COLOR_GUIDE,1,cv2.LINE_AA)
        draw_text_th(frame,"Guide: เดินตามเส้น 5 เมตร",(cx-100,40),color=COLOR_GUIDE)
    elif mode == 3:
        draw_text_th(frame,"Guide: เอื้อมมือแตะวงกลม",(20,40),color=COLOR_GUIDE)

def draw_info_panel(frame, mode, lean_lat,
                    balance_score, p_los, p_rms,
                    hip_rms, sh_rms, buf_secs,
                    step_count, gait_time, simple_sway,
                    reach_hit, w, h):
    pw = 290
    ph = 300 if mode == 1 else 195
    draw_panel(frame, 10, 10, pw, ph, COLOR_BG, 0.82)
    draw_text_bold(frame, f"Mode: {mode}", (20,38), 0.55, COLOR_TEAL)
    draw_text_th(frame, MODE_NAMES[mode], (20,62), 0.5, COLOR_WHITE)
    y = 90

    if mode == 1:
        if leaning_state == 1:
            draw_text_th(frame, "กรุณายืนนิ่งๆ รอสัญญาณ...", (20, y), color=COLOR_AMBER)
        elif leaning_state == 2:
            if countdown_start_t is not None:
                rem_cd = 3.0 - (time.time() - countdown_start_t)
                draw_text_bold(frame, f"เตรียมตัว! {max(1, int(np.ceil(rem_cd)))}", (20, y), 0.6, COLOR_ORANGE)
        elif leaning_state == 3:
            rem = max(0.0, LEANING_DURATION_SEC - leaning_time)
            timer_col = COLOR_AMBER if rem > 10 else COLOR_RED
            draw_text_th(frame, f"ทดสอบ: {rem:.1f} วินาที", (20, y), color=timer_col)
        elif final_score is not None:
            draw_text_th(frame, "เสร็จสิ้น", (20, y), color=COLOR_GREEN)
        else:
            draw_text_th(frame, "กด [SPACE] เริ่ม 60s", (20, y), color=COLOR_GRAY)

        display_score = final_score if final_score is not None else balance_score
        display_ar = final_ar_score if final_ar_score is not None else ar_score

        y += 24
        draw_text_th(frame, f"AR หยิบได้: {display_ar} ครั้ง/นาที", (20,y), color=COLOR_TEAL)
        y += 26

        BW = 245
        abs_lat  = abs(lean_lat)
        col_lat  = COLOR_RED if abs_lat > LOS_LATERAL else COLOR_GREEN
        dir_lat  = "ขวา" if lean_lat > 0 else "ซ้าย"
        draw_text_th(frame,
                     f"ซ้าย/ขวา: {lean_lat:+.1f}° ({dir_lat}) limit ±{LOS_LATERAL}°",
                     (20, y), color=col_lat)
        y += 19
        draw_bar(frame, 20, y, abs_lat, LOS_LATERAL*1.5, BW, 9, col_lat)
        draw_limit_marker(frame, 20+int(BW/1.5), y, BW, 9)
        y += 24

        rms_col = COLOR_GREEN if hip_rms < RMS_MAX*0.5 else \
                  COLOR_AMBER  if hip_rms < RMS_MAX     else COLOR_RED
        secs_str = f"{buf_secs:.0f}/{RMS_WINDOW_SEC:.0f}s"
        draw_text_th(frame,
                     f"RMS Hip:{hip_rms:.4f} ({secs_str})",
                     (20, y), 0.44, rms_col)
        y += 18
        draw_bar(frame, 20, y, hip_rms, RMS_MAX, BW, 9, rms_col)
        draw_limit_marker(frame, 20+BW, y, BW, 9)
        y += 24

        sc = score_color(display_score)
        draw_score_ring(frame, 48, y+25, display_score)
        draw_text_bold(frame, f"คะแนน: {display_score}/100", (90,y+16), 0.55, sc)
        draw_text_th(frame,   f"ระดับ: {score_label(display_score)}", (90,y+40), color=sc)

    elif mode == 2:
        timer_str = (f"{gait_time:.1f} วินาที"
                     if gait_running or gait_time > 0 else "กด [SPACE] เริ่ม")
        draw_text_th(frame, f"เวลา: {timer_str}", (20,y)); y+=26
        draw_text_th(frame, f"ก้าว: {step_count} ก้าว", (20,y), color=COLOR_AMBER); y+=26
        draw_text_th(frame, f"Trunk sway: {simple_sway:.1f}%", (20,y), color=COLOR_TEAL); y+=26
        status = "กำลังทดสอบ..." if gait_running else \
                 "เสร็จสิ้น" if gait_time > 0 else "พร้อมทดสอบ"
        draw_text_th(frame, f"สถานะ: {status}", (20,y))

    elif mode == 3:
        draw_text_th(frame, f"แตะได้: {len(reach_hit)}/{len(reach_targets)} เป้าหมาย",
                     (20,y), color=COLOR_GREEN); y+=26
        draw_text_th(frame, "กด [SPACE] สร้างเป้าใหม่", (20,y), color=COLOR_GRAY)

def draw_controls(frame, w):
    items = [("[1] โน้มตัว+AR", mode==1), ("[2] ก้าวเดิน", mode==2),
             ("[3] เอื้อมมือ", mode==3), ("[G] Guide", show_guide),
             ("[SPACE] Action", False), ("[Q] ออก", False)]
    px = w - 195
    draw_panel(frame, px-5, 10, 190, len(items)*26+16, COLOR_BG, 0.80)
    for i,(lbl,active) in enumerate(items):
        draw_text_th(frame, lbl, (px, 36+i*26),
                     color=COLOR_TEAL if active else COLOR_GRAY)

def main():
    global mode, show_guide, step_count, prev_ankle_y
    global gait_start, gait_running, gait_time
    global reach_targets, reach_hit, reach_spawn_t
    global leaning_state, ready_start_t, countdown_start_t, leaning_start, leaning_time, ar_score, ar_targets, final_score, final_ar_score

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่พบกล้อง")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    lean_lat = 0.0
    balance_score = 100
    p_los = p_rms = hip_rms = sh_rms = 0.0
    simple_sway = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        if show_guide:
            draw_guide(frame, mode, w, h)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=COLOR_TEAL, thickness=4, circle_radius=4),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=COLOR_WHITE, thickness=2))

            if mode == 1:
                lean_lat = calc_lean_lat(lms)
                update_sway_buffers(lms, now)
                hip_rms, sh_rms, rms_comb = calc_rms_sway()

                current_score = calc_balance_score(lean_lat, rms_comb)
                p_los = calc_los_penalty(lean_lat)
                p_rms = calc_rms_penalty(rms_comb)

                if leaning_state == 1:
                    # เช็คความนิ่ง ถ้า RMS ต่ำกว่าเกณฑ์ที่ 80% ของ MAX ค้างไว้ 1 วินาที
                    if hip_rms < RMS_MAX * 0.8:
                        if ready_start_t is None:
                            ready_start_t = now
                        elif now - ready_start_t >= 1.0:
                            leaning_state = 2
                            countdown_start_t = now
                    else:
                        ready_start_t = None
                    balance_score = current_score

                elif leaning_state == 2:
                    if countdown_start_t is not None and now - countdown_start_t >= 3.0:
                        leaning_state = 3
                        leaning_start = now
                        leaning_time = 0.0
                        ar_score = 0
                        ar_targets = [spawn_target(w, h) for _ in range(3)]
                        # เคลียร์ buffer ทิ้งก่อนเริ่มจับเวลาจริง
                        hip_buf.clear()
                        sh_buf.clear()
                    balance_score = current_score

                elif leaning_state == 3:
                    leaning_time = now - leaning_start
                    balance_score = current_score

                    hits = check_reach(lms, ar_targets, w, h)
                    if hits:
                        ar_score += len(hits)
                        ar_targets = [t for i, t in enumerate(ar_targets) if i not in hits]
                        while len(ar_targets) < 3:
                            ar_targets.append(spawn_target(w, h))

                    draw_reach_targets(frame, ar_targets, [], w, h)

                    for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        px, py = lm_px(lms[lm_id], w, h)
                        cv2.circle(frame,(px,py),12,COLOR_CORAL,-1)
                        cv2.circle(frame,(px,py),12,COLOR_WHITE,2)

                    if leaning_time >= LEANING_DURATION_SEC:
                        leaning_state = 0
                        final_score = balance_score
                        final_ar_score = ar_score
                        ar_targets = []
                        print(f"Test Complete. Score: {final_score}, AR: {final_ar_score}/min")
                else:
                    balance_score = current_score

                ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
                rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
                sh_mid = (int((ls.x+rs.x)/2*w), int((ls.y+rs.y)/2*h))
                hp_mid = (int((lh.x+rh.x)/2*w), int((lh.y+rh.y)/2*h))
                exceed = abs(lean_lat) > LOS_LATERAL
                lc = COLOR_RED if exceed else COLOR_GREEN
                cv2.line(frame, sh_mid, hp_mid, lc, 3, cv2.LINE_AA)
                cv2.circle(frame, sh_mid, 8, lc, -1)
                cv2.circle(frame, hp_mid, 8, lc, -1)

            elif mode == 2:
                detect_step(lms)
                simple_sway = calc_trunk_sway_simple(lms)
                if gait_running:
                    gait_time = now - gait_start
                for lm_id in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]:
                    px, py = lm_px(lms[lm_id], w, h)
                    cv2.circle(frame,(px,py),14,COLOR_AMBER,-1)
                    cv2.circle(frame,(px,py),14,COLOR_WHITE,2)

            elif mode == 3:
                if not reach_targets or now - reach_spawn_t > 4.0:
                    reach_targets = [spawn_target(w,h) for _ in range(3)]
                    reach_hit = []; reach_spawn_t = now
                for i in check_reach(lms, reach_targets, w, h):
                    if i not in reach_hit: reach_hit.append(i)
                draw_reach_targets(frame, reach_targets, reach_hit, w, h)
                for lm_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                    px, py = lm_px(lms[lm_id], w, h)
                    cv2.circle(frame,(px,py),12,COLOR_CORAL,-1)
                    cv2.circle(frame,(px,py),12,COLOR_WHITE,2)
        else:
            draw_text_bold(frame, "ไม่พบผู้ใช้", (w//2-100, h//2), color=COLOR_RED)
            hip_buf.clear(); sh_buf.clear()

        buf_secs = min(RMS_WINDOW_SEC, (now-hip_buf[0][2]) if hip_buf else 0.0)

        draw_info_panel(frame, mode, lean_lat,
                        balance_score, p_los, p_rms,
                        hip_rms, sh_rms, buf_secs,
                        step_count, gait_time, simple_sway,
                        reach_hit, w, h)
        draw_controls(frame, w)
        cv2.imshow("Balance Detection Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('1'):
            mode = 1
            leaning_state = 0
            final_score = None
            final_ar_score = None
            ar_score = 0
            ar_targets = []
            ready_start_t = None
            countdown_start_t = None
            hip_buf.clear()
            sh_buf.clear()
        elif key == ord('2'): mode=2
        elif key == ord('3'): mode=3; reach_targets=[]; reach_hit=[]
        elif key in (ord('g'),ord('G')): show_guide = not show_guide
        elif key == ord(' '):
            if mode == 1:
                if leaning_state == 0 or final_score is not None:
                    leaning_state = 1
                    ready_start_t = None
                    final_score = None
                    final_ar_score = None
                    ar_targets = []
            elif mode == 2:
                if not gait_running:
                    gait_start=now; gait_running=True
                    step_count=0; prev_ankle_y=None; gait_time=0.0
                else:
                    gait_running=False
            elif mode == 3:
                reach_targets=[spawn_target(w,h) for _ in range(3)]
                reach_hit=[]; reach_spawn_t=now

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()