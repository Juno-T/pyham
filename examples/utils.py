import os
from pathlib import Path
import numpy as np
import cv2

def render_frames(frames, outpath, fps=30, quality = "low"):
  assert outpath[-4:]==".mp4", "Not supported path"
  outpath=Path(outpath)
  os.makedirs(outpath.parent, exist_ok=True)
  size = frames[0].shape
  frameskip=1 # all frame
  if quality=="low":
    size = [s//2 for s in size]
    fps = fps//2
    frameskip=2 # every other frame
  out = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
  print(f"Rendering video with {len(frames)} frames of shape {frames[0].shape}")
  for frame in frames[::frameskip]:
    frame = np.asarray(frame)
    if frame.shape!=size:
      frame = cv2.resize(frame, (size[1], size[0]))
    # print(frame.mean(), frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
  out.release()
  print(f"Rendered video at {str(outpath)}")


def process_frames(frames, quality = "low"):
  """
    Process frames to wandb applicable format
  """
  size = frames[0].shape
  frameskip=1 # all frame
  if quality=="low":
    size = [size[0]//2, size[1]//2, size[2]]
    frameskip=2 # every other frame
  new_frames = [cv2.resize(frame, (size[1], size[0])) for frame in frames[::frameskip]]
  new_frames = np.array(new_frames)
  new_frames = np.transpose(new_frames, (0,3,1,2))
  return new_frames