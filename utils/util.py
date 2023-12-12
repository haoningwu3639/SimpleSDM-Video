import cv2
import numpy as np
from typing import List, Union
from PIL import Image
import inspect
import datetime
import copy
import sys
from typing import Dict, Union


def get_time_string() -> str:
    x = datetime.datetime.now()
    
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict

def export_to_video(video_frames: Union[List[np.ndarray], List[Image.Image]], output_video_path: str = None, fps: int = 8) -> str:
    # f, h, w, c
    if isinstance(video_frames[0], Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
        
    return output_video_path

def export_to_gif(video_frames, path, duration = 120, loop = 0, optimize = True):
    images = [Image.fromarray(frame, 'RGB') for frame in video_frames]

    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    
    return images
