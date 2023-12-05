import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as  tf
from re import DEBUG, sub
from flask import Flask, render_template, request, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
#from twilio.rest import Client
from ultralytics import YOLO
#import keys

# Khởi tạo Flask
app = Flask(__name__)
model = YOLO('best3.pt')

@app.route("/")
def hello_world():
    return render_template('index.html')
@app.route("/video_output")
def video_output():
    return render_template('video_output.html')
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files: # file la ten cua input de upload du lieu len
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            global mess
            mess = "Không phát hiện đám cháy"

            if file_extension == 'jpg': # neu file la dang .jpg
                frame = cv2.imread(filepath) # đọc ảnh
                #frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()

                #image = Image.open(io.BytesIO(frame))

                #yolo = YOLO('best.pt')

                results = model.predict(frame, save=True, stream=True)

                for result in results:
                    boxes = result.boxes.numpy()
                    if boxes:
                        #boxes = result.boxes.numpy()
                        for boxe in  boxes:
                            print(boxe.cls)
                            print(boxe.xyxy)
                            print(boxe.conf)
                            conf = boxe.conf*100
                            if conf >= 50:
                                mess = "Phát hiện đám cháy"
                                #sendsms()

                file_url = url_for('display', filename=f.filename)
                #return display(f.filename)

            elif file_extension == 'mp4':
                video_path = filepath # thay thế bằng đường dẫn video của bạn
                cap = cv2.VideoCapture(video_path) # để đọc video

                # lấy kích thước video
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #fps = int(cap.get(cv2.CAP_PROP_FPS))

                # xác định codec và tạo đối tượng VideoWriter
                #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # để viết video.
                #out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                #model = YOLO('best.pt')
                classNames = ["fire"]

                while cap.isOpened(): # # Đọc một khung hình từ video
                    success, frame = cap.read()
                    if not success:
                        break
                    else :
                        results = model(frame, save=True)
                        #results = model(frame, save=True)
                        print("ket qua ", results)
                        cv2.waitKey(1)

                        res_plotted = results[0].plot() ## Trực quan hóa kết quả trên khung
                        cv2.imshow("result yolov8", res_plotted)
                        video = "co video"
                        out.write(res_plotted)## ghi ra file output.mp4
                        for result in results:
                            boxes = result.boxes.numpy()
                            if boxes:
                                # boxes = result.boxes.numpy()
                                for boxe in boxes:
                                    print(boxe.cls)
                                    print(boxe.xyxy)
                                    print(boxe.conf)
                                    conf = boxe.conf * 100
                                    if conf >= 50:
                                        mess = "Phát hiện đám cháy"
                        if cv2.waitKey(1) == ord("q"):
                            break

                #cap.release()
                #cv2.destroyAllWindows()
                #return video_feed()
                return render_template("video_output.html", mess=mess)

    # folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    # image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
    return render_template('index.html', file_url=file_url, name=f.filename, mess=mess)

# def sendsms():
#     client = Client(keys.account_sid, keys.account_token)
#     message = client.messages.create(
#         from_='+13344630928',
#         to='+84888294121',
#         body="Phát hiện có đám cháy"
#     )

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ", directory) # runs/detect/predict20
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file) # ten image

    # runs/detect/predict20/image0.jpg
    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ)
    else:
        return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'D:/pythonProject/DetectionFire/output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)