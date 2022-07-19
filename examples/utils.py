import os
from pathlib import Path
import numpy as np
import cv2

def render_frames(frames, outpath, fps=30):
  assert outpath[-4:]==".mp4", "Not supported path"
  outpath=Path(outpath)
  os.makedirs(outpath.parent, exist_ok=True)
  size = frames[0].shape
  out = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
  print(f"Rendering video with {len(frames)} frames of shape {frames[0].shape}")
  for frame in frames:
    frame = np.asarray(frame)
    # print(frame.mean(), frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
  out.release()
  print(f"Rendered video at {str(outpath)}")