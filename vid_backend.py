import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
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
import tempfile
# import ffmpeg


"""
    TODO:
        
"""
class ChatGPTHandler:
    def __init__(self, api_key="sk-HYB9Yvv2lPFgP8MQE06024C8B26a4e41B144E2E6223e8949"):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def send_request(self, prompt):
        data = {
            "model": "gpt-3.5-turbo",
            "prompt": prompt,
            "max_tokens": 100
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        return response.json()
    
    

"""
this code used to isolate the Chat class from server
"""
class TimeChatBackend:
    args = None

    def __init__(self):
        print('Initializing Chat')
        self.args = self.parse_args()
        self.cfg = Config(self.args)
        self.MODEL_DIR = f"/data3/TimeChat-Jeff/ckpt/timechat/timechat_7b.pth"
        self.model_config = self.cfg.model_cfg
        self.model_config.device_8bit = self.args.gpu_id
        self.model_config.ckpt = self.MODEL_DIR
        self.model_cls = registry.get_model_class(self.model_config.arch)
        self.subprocess_export_berttokenizer_ssl()
        self.available_gpus = self.get_available_gpus()
        """
        TODO: fill account
        """
        self.ChatGPTHandler = ChatGPTHandler(api_key="")
        print(f"Available GPUs: {self.available_gpus}")
        while True:
            try:
                self.model = self.model_cls.from_config(self.model_config).to('cuda:{}'.format(1))
                self.model.eval()
                self.vis_processor_cfg = self.cfg.datasets_cfg.webvid.vis_processor.train
                self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(
                    self.vis_processor_cfg)
                # self.chat = Chat(self.model, self.vis_processor, device='cuda:{}'.format(self.args.gpu_id))
                self.chat = Chat(self.model, self.vis_processor, device='cuda:{}'.format(1))
                break
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f'CUDA out of memory on GPU {self.args.gpu_id}. Switching to another GPU.')
                    self.available_gpus.remove(self.args.gpu_id)
                    if len(self.available_gpus) == 0:
                        print('No more available GPUs. Exiting.')
                        raise e
                    self.args.gpu_id = self.available_gpus[0]
                    self.model_config.device_8bit = self.args.gpu_id
                else:
                    print('Failed to create chat')
                    raise e
            except Exception as e:
                print('Failed to create chat')
                raise e

    def get_available_gpus(self):
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                device = torch.device(f'cuda:{i}')
                torch.cuda.set_device(device)
                free_memory = torch.cuda.mem_get_info(device)[0] // (1024 ** 3)  # 获取可用显存，单位为GB
                available_gpus.append((i, free_memory))
            except:
                pass

        available_gpus.sort(key=lambda x: x[1], reverse=True)  # 按照可用显存从大到小排序
        print("Available GPUs sorted by free memory:")
        for gpu_id, free_memory in available_gpus:
            print(f"GPU {gpu_id}: {free_memory} GB")

        return [gpu_id for gpu_id, _ in available_gpus]

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--num-beams", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--text-query", default="What is he doing?", help="question the video")
        parser.add_argument("--video-path", default='videos/video_20240504060927.mp4', help="path to video file.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
                 "in xxx=yyy format will be merged into config file (deprecate), "
                 "change to --cfg-options instead.",
        )
        args = parser.parse_args(args=[])
        return args

    def subprocess_export_berttokenizer_ssl(self):
        command = "echo $HF_ENDPOINT"
        result = subprocess.run(command, shell=True, check=True, text=True,
                                env={"HF_ENDPOINT": "https://hf-mirror.com"})
        print("export huggingface bert tokenizer successed!")
    
  
    def video_loader(self, video_path,
                     n_frms: int,
                     sampling: str,
                     return_msg=True):
        # return load_video(self.args.video_path,32,"uniform",True)
        return load_video(video_path=video_path, n_frms=n_frms, sampling='uniform', return_msg=True)

    def setting_msg_without_audio(self, video_path: str, n_frms: int, sys_prompt: str):
        img_list = []
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = sys_prompt
        img_list, msg = self.chat.upload_video_without_audio_backend(video_path=video_path, conv=chat_state,
                                                                     img_list=img_list, n_frms=n_frms)
        return chat_state, img_list, msg

    def get_answer(self, anser_prompt: str,
                   chat_state,
                   img_list,
                   args):
        num_beams = self.args.num_beams
        temperature = self.args.temperature
        llm_message = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=300,
            max_length=2000
        )[0]
        return llm_message

    def understand_video(self, video_path: str,
                         load_video_n_frms: int,
                         upload_video_n_frms: int,
                         sys_prompt: str,
                         anser_prompt: str):
        video, _ = self.video_loader(video_path=video_path, n_frms=load_video_n_frms, sampling='uniform',
                                     return_msg=True)
        chat_state, img_list, _ = self.setting_msg_without_audio(video_path=video_path, n_frms=upload_video_n_frms,
                                                                 sys_prompt=sys_prompt)
        llm_message = self.get_answer(anser_prompt=anser_prompt, chat_state=chat_state, img_list=img_list,
                                      args=self.args)
        # print(llm_message)
        return llm_message
    
    """
        ChatGPT的image understanding API
    """
    def understand_image(self,
                     image_path:str,
                    #  sys_prompt:str,
                     answer_prompt:str,
                     return_msg=True):
        return self.ChatGPTHandler.send_request(prompt)
    
    
    def test(self):
        img_list = []
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        msg = self.chat.upload_video_without_audio(
            video_path=self.args.video_path,
            conv=chat_state,
            img_list=img_list,
            n_frms=96,
        )
        text_input = "帮我总结视频中发生的内容"
        print(text_input)

        self.chat.ask(text_input, chat_state)

        num_beams = self.args.num_beams
        temperature = self.args.temperature
        llm_message = self.chat.answer(conv=chat_state,
                                       img_list=img_list,
                                       num_beams=num_beams,
                                       temperature=temperature,
                                       max_new_tokens=300,
                                       max_length=2000)[0]

        print(llm_message)


if __name__ == '__main__':
    tb = TimeChatBackend()
    tb.understand_video(
        video_path='/data/caitiefu/TimeChat/videos/video_20240504060927.mp4',
        load_video_n_frms=32,
        upload_video_n_frms=96,
        sys_prompt="You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail.",
        anser_prompt="帮我总结视频中发生的内容"
    )
    # load_video('examples/jinhui_test.mp4',32,'uniform',True)
    # tb.test()
