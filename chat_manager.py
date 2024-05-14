import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr
import os
import tempfile
import ffmpeg
import cv2
import numpy as np
import base64
from io import BytesIO
import tempfile
import os
from PIL import Image
import datetime
import subprocess

    
    
class NewMP4Handler(FileSystemEventHandler):
    def __init__(self, mp4_list):
        self.mp4_list = mp4_list

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            self.mp4_list.append(event.src_path)
            print(f"New MP4 file detected: {event.src_path}")

class MP4Queue:
    def __init__(self, path):
        self.path = path
        self.mp4_list = []
        self.current_index = 0
        self.observer = Observer()
        self.event_handler = NewMP4Handler(self.mp4_list)
        self.observer.schedule(self.event_handler, self.path, recursive=False)
        self.observer.start()

    def get_next_video(self):
        if self.current_index < len(self.mp4_list):
            video_path = self.mp4_list[self.current_index]
            self.current_index += 1
            return video_path
        return None

    def remove_processed_video(self, video_path):
        try:
            self.mp4_list.remove(video_path)
            self.current_index -= 1
            print(f"Removed processed video: {video_path}")
        except ValueError:
            print(f"Video not found in queue: {video_path}")

    def stop(self):
        self.observer.stop()
        self.observer.join()
        
class ImageHandler(FileSystemEventHandler):
    def __init__(self, image_list):
        self.image_list = image_list

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            self.image_list.append(event.src_path)
            print(f"New Image file detected: {event.src_path}")
        
class IMAGEQueue:
    def __init__(self, path):
        self.path = path
        self.image_list = []
        self.current_index = 0
        self.observer = Observer()
        self.event_handler = NewMP4Handler(self.image_list)
        self.observer.schedule(self.event_handler, self.path, recursive=False)
        self.observer.start()

    def get_next_image(self):
        if self.current_index < len(self.image_list):
            image_path = self.image_list[self.current_index]
            self.current_index += 1
            return image_path
        return None

    def remove_processed_video(self, image_path):
        try:
            self.image_list.remove(image_path)
            self.current_index -= 1
            print(f"Removed processed video: {image_path}")
        except ValueError:
            print(f"Image not found in queue: {image_path}")

    def stop(self):
        self.observer.stop()
        self.observer.join()

class QAQueue:
            
        

class ChatManager:
    def __init__(self, output_dir='videos'):
        self.current_frame = None
        self.analysis_result = ""
        self.output_dir = output_dir
        self.video_count = 0
        self.frame_count = 0
        self.frames = []
        self.max_frames = 100
        self.output_h264_dir = 'h264_frames'
        
        # 创建输出目录(如果不存在)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_h264_dir, exist_ok=True)
    
    """
    视频临时文件预处理
    """
    def process_video(self,data,image):
        # 假设data是base64编码的图像数据
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))
        temp_file = BytesIO()
        image.save(temp_file, format='JPEG')  # 将Bitmap转为JPEG并保存在内存中
        images.append(temp_file.getvalue())   # 存储JPEG数据

        # 当收集到足够的图像帧时，创建视频
        if len(images) == 125:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # 保存图像到临时目录
                for idx, img_data in enumerate(images):
                    img_path = os.path.join(tmpdirname, f'image_{idx:03d}.jpeg')
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_data)

                # 使用ffmpeg创建视频
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                output_video_path = f'/home/tiefucai/TimeChat/videos/video_{timestamp}.mp4'
                output_directory = os.path.dirname(output_video_path)

                # 检查目录是否存在，不存在则创建
                # if not os.path.exists(output_directory):
                #     os.makedirs(output_directory)

                # 构建ffmpeg命令
                command = [
                    'ffmpeg', 
                    '-framerate', '25', 
                    '-i', f'{tmpdirname}/image_%03d.jpeg',
                    '-c:v', 'libx264', 
                    '-pix_fmt', 'yuv420p', 
                    output_video_path
                ]

                # 使用subprocess运行ffmpeg命令
                subprocess.run(command, check=True)
            # 清空images列表以备下次使用
            temp_images = images
            images = []
            return temp_images
        return False
    
    """
    图片临时文件预处理
    """
    def process_image(self,image):
         # 假设data是base64编码的图像数据
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))
        temp_file = BytesIO()
        image.save(temp_file, format='JPEG')  # 将Bitmap转为JPEG并保存在内存中
        # images.append(temp_file.getvalue())
        return temp_file.getvalue()

  