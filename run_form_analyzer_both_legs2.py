#!/usr/bin/env python3
"""
Running form analyzer (side view).

Reads a webcam or a video file, runs MediaPipe Pose on every frame, draws the
skeleton over the runner, measures joint angles live for BOTH legs, detects
foot strikes to get cadence / landing metrics, and prints coaching tips.

"Main" leg = the leg nearest the camera (the one MediaPipe sees best).
"Back" leg = the other leg, partly hidden behind the main one, so its numbers
are noisier.

Install:
    pip install mediapipe opencv-python numpy scipy
    (If you get "module 'mediapipe' has no attribute 'solutions'", use:
     pip install "mediapipe==0.10.14")

Usage:
    python run_form_analyzer_both_legs.py --source 0                        # webcam
    python run_form_analyzer_both_legs.py --source run.mp4                  # video file
    python run_form_analyzer_both_legs.py --source run.mp4 --out overlay.mp4 --csv angles.csv --height-cm 175
    python run_form_analyzer_both_legs.py --source run.mp4 --min-vis 0.3    # keep the back leg more often

By default the per-frame angles are saved to angles_both.csv and the summary to report_both.txt
(change them with --csv and --report). Add --runner "Name" to label the run: the name goes into the
report and CSV, and the files become angles_name.csv and report_name.txt.

CSV: one row per frame with separate columns for each leg:
    t, main_side, main_ankle_deg, main_knee_deg, main_foot_deg, main_shank_deg,
    back_side, back_ankle_deg, back_knee_deg, back_foot_deg, back_shank_deg, trunk_lean_deg
A leg's cells are blank on frames where it wasn't visible enough.

Keys while the window is open:  q = quit

Filming tips: camera side-on at hip height, whole body in frame, camera fixed
(tripod), 60 fps or more if your camera allows, bright light, fitted clothes.
"""

import argparse
import csv
import math
import re
import time

import numpy as np
from scipy.signal import find_peaks, savgol_filter

KEYS = ("shoulder", "hip", "knee", "ankle", "heel", "toe")
BGR_TEAL, BGR_AMBER, BGR_WHITE = (212, 234, 94), (32, 176, 255), (255, 255, 255)
BGR_GREEN = (60, 220, 60)
BGR_FAR = (255, 150, 120)      # back leg: light blue
CSV_FIELDS = [
    "t", "runner",
    "main_side", "main_ankle_deg", "main_knee_deg", "main_foot_deg", "main_shank_deg", "main_hip_flexion_deg",
    "back_side", "back_ankle_deg", "back_knee_deg", "back_foot_deg", "back_shank_deg", "back_hip_flexion_deg",
    "trunk_lean_deg",
]

# ----------------------------------------------------------------------------
# Geometry (image coordinates: x to the right, y DOWN). `facing` is +1 if the
# runner faces right in the image, -1 if they face left.
# ----------------------------------------------------------------------------
def angle_at(a, b, c):
    """Interior angle in degrees at point b between segments b->a and b->c."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / denom, -1, 1)))


def foot_angle(heel, toe, facing):
    """Foot angle to the ground in degrees. Positive = toe higher than heel
    (heel-strike posture), negative = toe pointing down (forefoot)."""
    dx = (toe[0] - heel[0]) * facing
    dy = heel[1] - toe[1]
    return math.degrees(math.atan2(dy, dx))


def trunk_lean(shoulder, hip, facing):
    """Trunk lean from vertical in degrees. Positive = leaning forward."""
    fwd = (shoulder[0] - hip[0]) * facing
    up = hip[1] - shoulder[1]
    return math.degrees(math.atan2(fwd, up))


def shank_angle(knee, ankle, facing):
    """Shank (shin) angle from vertical. Positive = foot ahead of the knee."""
    fwd = (ankle[0] - knee[0]) * facing
    down = ankle[1] - knee[1]
    return math.degrees(math.atan2(fwd, down))


def hip_flexion(shoulder, hip, knee, facing):
    """Hip flexion: angle between the trunk (extended downward) and the thigh.
    0 = standing straight, 90 = knee level with the hip, negative = leg behind the body."""
    d = (hip[0] - shoulder[0], hip[1] - shoulder[1])
    v = (knee[0] - hip[0], knee[1] - hip[1])
    return -facing * math.degrees(math.atan2(d[0] * v[1] - d[1] * v[0], d[0] * v[0] + d[1] * v[1]))


def fill_nan(arr):
    """Linearly interpolate NaN gaps in an (n, 2) array."""
    out = arr.copy()
    idx = np.arange(len(out))
    for c in range(out.shape[1]):
        ok = ~np.isnan(out[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(idx, idx[ok], out[ok, c])
    return out


# ----------------------------------------------------------------------------
# Recording and analysis
# ----------------------------------------------------------------------------
class Recorder:
    """Stores landmark positions (pixels) for both legs, frame by frame."""

    def __init__(self):
        self.t = []
        self.p = {s: {k: [] for k in KEYS} for s in ("left", "right")}

    def add(self, t, pts):
        """pts = {'left': {key: (x, y)} or None, 'right': ...}"""
        self.t.append(t)
        for s in self.p:
            for k in KEYS:
                self.p[s][k].append(pts[s][k] if pts.get(s) else (np.nan, np.nan))

    def arrays(self, n=None):
        sl = slice(-n, None) if n else slice(None)
        t = np.array(self.t[sl], float)
        P = {s: {k: fill_nan(np.array(self.p[s][k][sl], float)) for k in KEYS} for s in self.p}
        return t, P


def analyze(t, P, fps, height_cm=None):
    """Detect foot strikes and compute stride metrics. Returns a dict or None.

    Foot strike is approximated as the moment the ankle is furthest FORWARD of
    the hip; toe-off as furthest BACK. That works well for a fixed side-on
    camera (treadmill) and needs no force plate.
    """
    if len(t) < fps * 2:
        return None
    # drop any leg that is still missing data after gap-filling
    P = {s: Q for s, Q in P.items() if all(np.isfinite(Q[k]).all() for k in KEYS)}
    if not P:
        return None
    toe_minus_heel = np.concatenate([P[s]["toe"][:, 0] - P[s]["heel"][:, 0] for s in P])
    facing = 1.0 if np.nanmedian(toe_minus_heel) >= 0 else -1.0
    leg_px = float(np.nanmedian(np.concatenate([
        np.linalg.norm(P[s]["hip"] - P[s]["knee"], axis=1) + np.linalg.norm(P[s]["knee"] - P[s]["ankle"], axis=1)
        for s in P])))
    if not np.isfinite(leg_px) or leg_px < 1:
        return None

    events = []
    for side, Q in P.items():
        s = (Q["ankle"][:, 0] - Q["hip"][:, 0]) * facing
        w = max(5, (int(fps * 0.12) // 2) * 2 + 1)
        if len(s) > w:
            s = savgol_filter(s, w, 2)
        swing = float(np.percentile(s, 95) - np.percentile(s, 5))      # adapt to how far the ankle really travels
        dist, prom = max(2, int(fps * 0.28)), max(0.06 * leg_px, 0.3 * swing)
        strikes, _ = find_peaks(s, distance=dist, prominence=prom)
        toeoffs, _ = find_peaks(-s, distance=dist, prominence=prom)
        for i in strikes:
            later = toeoffs[toeoffs > i]
            contact = t[later[0]] - t[i] if len(later) and t[later[0]] - t[i] < 0.6 else float("nan")
            events.append(dict(
                i=int(i), t=float(t[i]), side=side, over_px=float(s[i]), contact=float(contact),
                ankle=angle_at(Q["knee"][i], Q["ankle"][i], Q["toe"][i]),
                foot=foot_angle(Q["heel"][i], Q["toe"][i], facing),
                knee=angle_at(Q["hip"][i], Q["knee"][i], Q["ankle"][i]),
                lean=trunk_lean(Q["shoulder"][i], Q["hip"][i], facing),
                shank=shank_angle(Q["knee"][i], Q["ankle"][i], facing)))     # <- was missing
    events.sort(key=lambda e: e["t"])
    if len(events) < 3:
        return None

    # Peak hip flexion = how high the knee comes up in each stride (90 = knee level with the hip)
    flex_all = []
    for side, Q in P.items():
        d, v = Q["hip"] - Q["shoulder"], Q["knee"] - Q["hip"]
        series = -facing * np.degrees(np.arctan2(d[:, 0] * v[:, 1] - d[:, 1] * v[:, 0], (d * v).sum(1)))
        w2 = max(5, (int(fps * 0.1) // 2) * 2 + 1)
        if len(series) > w2:
            series = savgol_filter(series, w2, 2)
        idx = [e["i"] for e in events if e["side"] == side]
        flex_all += [float(series[a_:b_ + 1].max()) for a_, b_ in zip(idx[:-1], idx[1:]) if b_ - a_ >= 2]

    med = lambda key: float(np.nanmedian([e[key] for e in events]))
    span = events[-1]["t"] - events[0]["t"]
    cm_per_px = (0.53 * height_cm) / leg_px if height_cm else None   # leg length ~ 53% of height
    over_px = med("over_px")
    return dict(
        steps=len(events),
        cadence=(len(events) - 1) / span * 60 * (2 / len(P)) if span > 0 else float("nan"),
        contact_ms=med("contact") * 1000,
        over_pct_leg=over_px / leg_px * 100,
        over_cm=over_px * cm_per_px if cm_per_px else None,
        foot=med("foot"), ankle=med("ankle"), knee=med("knee"), lean=med("lean"),
        shank=med("shank"),
        hip_flex=float(np.median(flex_all)) if flex_all else None,
        fps=fps, last=events[-1])


def coach(m):
    """Rule-based tips. Thresholds are illustrative coaching ranges, not clinical."""
    tips = []
    over = m["over_cm"] if m["over_cm"] is not None else None
    overstriding = (over is not None and over > 15) or (over is None and m["over_pct_leg"] > 16)
    if m["cadence"] < 168:
        tips.append(f"Quicken your cadence: {m['cadence']:.0f} steps/min. Try a metronome at ~172 bpm "
                    "for the first five minutes of easy runs; quicker, shorter steps cut overstriding.")
    if overstriding:
        d = f"{over:.0f} cm" if over is not None else f"{m['over_pct_leg']:.0f}% of leg length"
        tips.append(f"Overstriding: your foot lands {d} ahead of your hips. Think 'quick feet under your hips'.")
    behind = (over is not None and over < -5) or (over is None and m["over_pct_leg"] < -6)
    if behind:
        d = f"{abs(over):.0f} cm" if over is not None else f"{abs(m['over_pct_leg']):.0f}% of leg length"
        tips.append(f"Your foot lands {d} behind your hips. Aim to land under your body, directly beneath your hips. "
                    "(Check the overlay video that landings are being picked at the right moment.)")
    if overstriding or behind:
        for k, t_ in enumerate(tips):
            if t_.startswith("Quicken your cadence"):
                tips[k] = t_ + " Also aim to land under your body, with your foot close to directly beneath your hips."
    if m["foot"] > 14 and overstriding:
        tips.append(f"Toes are up {m['foot']:.0f} degrees at landing while reaching out front. "
                    "Land softer and flatter under your body; don't force a forefoot strike.")
    if m["lean"] < 4:
        tips.append(f"Lean forward slightly: trunk lean is {m['lean']:.1f} degrees. Lean from the ankles, "
                    "not the waist, like a string is pulling your chest forward.")
    elif m["lean"] > 10:
        tips.append(f"Stand up taller: trunk lean is {m['lean']:.1f} degrees. Run tall, as if a string "
                    "pulls the top of your head upward.")
    if m["fps"] >= 50 and m["contact_ms"] > 260:
        tips.append(f"Ground contact is {m['contact_ms']:.0f} ms. Add four relaxed 20-second strides "
                    "after easy runs, aiming for light, quick contacts.")
    if m.get("hip_flex") is not None and m["hip_flex"] < 30:
        tips.append(f"Drive the knee up more: your hip flexes {m['hip_flex']:.0f} degrees at the top of each step "
                    "(90 = knee level with the hip). Try A-skips or high-knee drills before runs. "
                    "(It grows with speed, so slow jogging or running in place reads lower.)")
    return tips or ["Everything sits inside the target bands. Nice, steady form."]


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def put_text(img, text, org, color=BGR_WHITE, scale=0.6, thick=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


BGR_PANEL = (28, 24, 20)

def draw_hud(img, lines):
    """Translucent dark panel with a teal accent bar, top-left. First line is the header."""
    import cv2
    font, scale, pad, lh = cv2.FONT_HERSHEY_DUPLEX, 0.55, 14, 28
    w = max(cv2.getTextSize(l, font, scale, 1)[0][0] for l in lines) + pad * 2 + 8
    h = lh * len(lines) + pad
    x0, y0 = 16, 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), BGR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + 4, y0 + h), BGR_TEAL, -1)
    for n, line in enumerate(lines):
        cv2.putText(img, line, (x0 + pad + 8, y0 + pad + 12 + lh * n),
                    font, scale, BGR_WHITE, 1, cv2.LINE_AA)

def draw_angle(img, a, b, c, label, color=BGR_AMBER, radius=30, dy=5):
    """Arc at joint b between b->a and b->c, with a text label."""
    import cv2
    a1 = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    a2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
    d = (a2 - a1 + 180) % 360 - 180
    start, end = (a1, a1 + d) if d >= 0 else (a1 + d, a1)
    cv2.ellipse(img, (int(b[0]), int(b[1])), (radius, radius), 0, start, end, color, 2, cv2.LINE_AA)
    put_text(img, label, (int(b[0]) + radius + 6, int(b[1]) + dy), color)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(args):
    import cv2
    import mediapipe as mp

    mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
    L = mp_pose.PoseLandmark
    SIDE_IDX = {
        s: dict(shoulder=getattr(L, f"{s.upper()}_SHOULDER"), hip=getattr(L, f"{s.upper()}_HIP"),
                knee=getattr(L, f"{s.upper()}_KNEE"), ankle=getattr(L, f"{s.upper()}_ANKLE"),
                heel=getattr(L, f"{s.upper()}_HEEL"), toe=getattr(L, f"{s.upper()}_FOOT_INDEX"))
        for s in ("left", "right")}

    is_cam = str(args.source).isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)) if args.out else None

    pose = mp_pose.Pose(model_complexity=args.complexity, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    rec, rows = Recorder(), []
    facing_est, side_score, side_lock = 0.0, {"left": 0.0, "right": 0.0}, None
    summary, frame_i, t0 = None, 0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.perf_counter() - t0) if is_cam else frame_i / fps
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            pts, vis = {}, {}
            for s, idx in SIDE_IDX.items():
                vis[s] = min(lm[i].visibility for i in idx.values())
                pts[s] = {k: (lm[i].x * fw, lm[i].y * fh) for k, i in idx.items()} if vis[s] >= args.min_vis else None
            rec.add(t, pts)

            # running estimate of which way the runner faces (toe vs heel)
            dxs = [p["toe"][0] - p["heel"][0] for p in pts.values() if p]
            if dxs:
                facing_est = 0.95 * facing_est + 0.05 * float(np.mean(dxs))
            facing = 1.0 if facing_est >= 0 else -1.0

            # main leg = nearest the camera (most visible); lock it after 30 frames
            for s in side_score:
                side_score[s] += vis[s]
            if args.side != "auto":
                side_lock = args.side
            elif side_lock is None and frame_i >= 30:
                side_lock = max(side_score, key=side_score.get)
            side = side_lock or max(side_score, key=side_score.get)
            back_side = "left" if side == "right" else "right"

            # skeleton over the person
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=BGR_GREEN, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=BGR_GREEN, thickness=2))

            # measure + draw BOTH legs; back leg first so the main leg sits on top
            row = dict.fromkeys(CSV_FIELDS)
            row.update(t=round(t, 4), runner=args.runner or None, main_side=side, back_side=back_side)
            lean_val = None
            for s, who, color, y_text in ((back_side, "back", BGR_FAR, fh - 44), (side, "main", BGR_AMBER, fh - 16)):
                p = pts[s]
                if not p:
                    continue
                for a, b in (("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "toe"), ("toe", "ankle")):
                    cv2.line(frame, tuple(map(int, p[a])), tuple(map(int, p[b])), color, 4 if who == "main" else 3, cv2.LINE_AA)
                ankle_deg = angle_at(p["knee"], p["ankle"], p["toe"])
                knee_deg = angle_at(p["hip"], p["knee"], p["ankle"])
                foot_deg = foot_angle(p["heel"], p["toe"], facing)
                shank_deg = shank_angle(p["knee"], p["ankle"], facing)
                hip_deg = hip_flexion(p["shoulder"], p["hip"], p["knee"], facing)
                if who == "main" or lean_val is None:          # trunk lean: main leg's reading wins
                    lean_val = trunk_lean(p["shoulder"], p["hip"], facing)
                pre = "" if who == "main" else "Back "
                dy = 5 if who == "main" else -14
                draw_angle(frame, p["knee"], p["ankle"], p["toe"], f"{pre}ankle {ankle_deg:.0f}", color, 30, dy)
                draw_angle(frame, p["hip"], p["knee"], p["ankle"], f"{pre}knee {knee_deg:.0f}",
                           BGR_WHITE if who == "main" else color, 24, dy)
                row[f"{who}_ankle_deg"] = round(ankle_deg, 1)
                row[f"{who}_knee_deg"] = round(knee_deg, 1)
                row[f"{who}_foot_deg"] = round(foot_deg, 1)
                row[f"{who}_shank_deg"] = round(shank_deg, 1)
                row[f"{who}_hip_flexion_deg"] = round(hip_deg, 1)
                below = (2 * p["hip"][0] - p["shoulder"][0], 2 * p["hip"][1] - p["shoulder"][1])     # trunk line continued downward
                draw_angle(frame, below, p["hip"], p["knee"], f"{pre}hip {hip_deg:.0f}", color, 40 if who == "main" else 52, dy)
                put_text(frame, f"{who.capitalize()} ({s}): hip {hip_deg:.0f}   shin {shank_deg:.0f}   ankle {ankle_deg:.0f}",
                         (14, y_text), color)
            if lean_val is not None:
                row["trunk_lean_deg"] = round(lean_val, 1)
                rows.append(row)

        # refresh rolling metrics from the last ~8 seconds
        if frame_i % 15 == 0 and frame_i > fps * 3:
            t_arr, P = rec.arrays(int(fps * 8))
            summary = analyze(t_arr, P, fps, args.height_cm) or summary
        hud = ["COLLECTING STEPS..."] if summary is None else [
            "LIVE METRICS",
            f"Cadence  {summary['cadence']:.0f} steps/min",
            f"Foot {summary['last']['foot']:.0f} deg   Ankle {summary['last']['ankle']:.0f} deg"]
        draw_hud(frame, hud)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Running form analyzer (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_i += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ---- final report over the whole recording
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved per-frame angles (main + back leg) to {args.csv}")
    t_arr, P = rec.arrays()
    final = analyze(t_arr, P, fps, args.height_cm)
    if not final:
        print("Not enough steps detected. Make sure the runner is side-on, fully in frame, and running for 5+ seconds.")
        return
    lines = [
        f"--- Summary ({final['steps']} steps, {fps:.0f} fps) ---",
        f"Cadence:            {final['cadence']:.0f} steps/min",
        f"Foot lands ahead:   {final['over_pct_leg']:.0f}% of leg length" +
        (f" (~{final['over_cm']:.0f} cm)" if final["over_cm"] is not None else "  (add --height-cm for cm)"),
        f"Foot angle at land: {final['foot']:.0f} deg   Ankle angle at land: {final['ankle']:.0f} deg",
        f"Knee at landing:    {final['knee']:.0f} deg   Trunk lean: {final['lean']:.1f} deg",
        f"Shank angle (shin angle) at landing: {final['shank']:.0f} deg   (0 = shin vertical, + = foot ahead of knee)",
        (f"Hip flexion:        {final['hip_flex']:.0f} deg   (peak at the top of each step; 90 = knee level with hip)"
         if final["hip_flex"] is not None else "Hip flexion:        not enough strides to measure"),
        f"Ground contact:     {final['contact_ms']:.0f} ms" + ("" if fps >= 50 else "  (unreliable below ~60 fps)"),
        "",
        "Coach notes:",
    ] + [f" - {tip}" for tip in coach(final)]
    if args.runner:
        lines.insert(1, f"Runner: {args.runner}")
    print("\n" + "\n".join(lines))
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSaved report to {args.report}")


def main():
    ap = argparse.ArgumentParser(description="Side-view running form analyzer")
    ap.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    ap.add_argument("--out", help="save the overlay video here (mp4)")
    ap.add_argument("--csv", help="save per-frame angles here (default: angles_both.csv, or angles_<runner>.csv)")
    ap.add_argument("--side", choices=["auto", "left", "right"], default="auto", help="which leg is the main leg")
    ap.add_argument("--height-cm", type=float, help="runner height, to convert pixels to centimetres")
    ap.add_argument("--complexity", type=int, choices=[0, 1, 2], default=1, help="MediaPipe model size (2 = most accurate)")
    ap.add_argument("--min-vis", type=float, default=0.5, help="ignore a leg when landmark visibility is below this")
    ap.add_argument("--cam-fps", type=int, default=60, help="requested webcam frame rate")
    ap.add_argument("--no-display", action="store_true", help="don't open a preview window")
    ap.add_argument("--report", help="save the summary and coach notes here (default: report_both.txt, or report_<runner>.txt)")
    ap.add_argument("--runner", help="runner / person name; added to the report and CSV and used in the default file names")
    args = ap.parse_args()
    slug = re.sub(r"[^a-z0-9]+", "_", args.runner.lower()).strip("_") if args.runner else ""
    args.csv = args.csv or (f"angles_{slug}.csv" if slug else "angles_both.csv")
    args.report = args.report or (f"report_{slug}.txt" if slug else "report_both.txt")
    run(args)


if __name__ == "__main__":
    main()
