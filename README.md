# RunningForm Analyzer/AI (browser version)

One file, `index.html`. The pose tracking, angle measurements, report and overlay video all run inside the visitor's
browser. There is no server and nothing is uploaded, so it can be hosted for free as a static website.

## What it does

- Tracks the runner with MediaPipe Pose Landmarker (the same 33-point body model as the Python version).
- Measures cadence, ground contact, foot landing distance, trunk lean, hip flexion (how high the knee goes),
  foot and shin angles, at landing and push-off, for both legs. Writes the coach notes.
- Plays your video with the skeleton and hip, knee and ankle angles drawn on it. Slow motion (0.25x, 0.5x, 1x),
  previous/next frame buttons, a scrubber, arrow keys, and toggles for the back leg, joint angles and skeleton.
- Save video (the overlay baked in, recorded in real time), Save report (an HTML file you can print to PDF),
  and a CSV of the angles for every frame.

## Good to know

- First run downloads the pose library (from cdn.jsdelivr.net) and the model (from storage.googleapis.com), roughly
  10 to 30 MB. Those servers receive no video. Fully offline use would need the files self-hosted.
- Time: it steps through the clip frame by frame, so a 12-second clip takes about a minute or two, depending on the device.
  Only the first 90 seconds of a video are analyzed. Keep the tab open and in front while it runs.
- Save video records in real time, so a 30-second clip takes about 30 seconds to save. The file type depends on
  the browser (MP4 in Safari and recent Chrome, WebM in some others).
- The browser has to be able to open the video. iPhone .mov files open in Safari; other browsers may not read them.
- Results are close to the Python version's, not identical, because the pose model package differs.
- Settings you can change are at the top of the script in `index.html`: `MP_VERSIONS` (pose library versions to try),
  `MODEL_URL`, `MAX_DIM`, `MAX_SECONDS`.
