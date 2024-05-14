from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
from chat_manager import ChatManager,MP4Queue,NewMP4Handler
from vid_backend import  TimeChatBackend
import logging
import os
import time
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*",ping_interval=2000,ping_timeout=120000,async_mode='threading')
chat_manager = ChatManager()
tb = TimeChatBackend()
# directory = '/data3/TimeChat-Jeff/videos'
# 存储临时图像数据的列表
image_list = []
video_list = []
mp4_queue = MP4Queue('/data3/TimeChat-Jeff/videos')
image_queue = IMAGEQueue('/data3/TimeChat-Jeff/image')
last_video = None
last_image = None


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image_understand')
def understand_image(data):
    global image_list
    image_list = chat_manager.process_image(data,image_list)
    image_path = image_queue.get_next_image()
    try:
        if image_path != None:
            answer = tb.understand_image(
                image_path = image_path,
                answer_prompt="帮我总结视频中发生的内容。"
            )
            emit('respone',{"respone":answer})
    except:
        image_queue.current_index -= 1
        emit('respone',{"status":"No iamge tp process"})


@socketio.on('video_record')
def handle_video_record(data):
    global video_list
    video_list = chat_manager.process_video(data, video_list)
    video_path = mp4_queue.get_next_video()
    try:
        if video_path != None:
            answer = tb.understand_video(
                    video_path = video_path,
                    load_video_n_frms= 32,
                    upload_video_n_frms= 96,
                    sys_prompt= "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail.",
                    anser_prompt= "帮我总结视频中发生的内容"
                )
            emit('response', {"response": answer})
    except:
        mp4_queue.current_index -= 1
        emit('response', {"status": "No video to process"})


def check_and_update_files(directory, tracked_files):
    current_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    new_files = [f for f in current_files if f not in tracked_files]
    if new_files:
        for f in new_files:
            tracked_files.append(f)

@app.route('/video_record', methods=['POST'])
def handle_video_record():
    global images
    data = request.json['data']
    images = chat_manager.process_video(data, images)
    video_path = mp4_queue.get_next_video()
    try:
        if video_path != None:
            answer = tb.understand_video(
                    video_path = video_path,
                    load_video_n_frms= 32,
                    upload_video_n_frms= 96,
                    sys_prompt= "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail.",
                    anser_prompt= "帮我总结视频中发生的内容"
                )
            return jsonify({"response": answer})
    except:
        mp4_queue.current_index -= 1
        return jsonify({"status": "No video to process"})
    # video_path = mp4_queue.get_next_video()
    return jsonify({"status": "success but not yet processed"})


@app.route('/get_analysis_result', methods=['GET'])
def get_analysis_result():
    return chat_manager.get_analysis_result()


@app.route('/init_cfg', methods=['GET'])
def init_cfg():
    return chat_manager.init_cfg()


@app.route('/init_chat', methods=['GET'])
def init_chat():
    return chat_manager.init_chat()


@app.route('/loading_video', methods=['GET'])
def loading_video():
    return chat_manager.loading_video()


@app.route('/text_input', methods=['GET'])
def text_input():
    return chat_manager.text_input()


@app.route('/answer', methods=['GET'])
def answer():
    return chat_manager.answer()


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', port=5000)
    app.run( host='0.0.0.0', port=5000)
