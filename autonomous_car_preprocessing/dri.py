import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


sio = socketio.Server()
app = Flask(__name__)

@app.route('/home')
def greeting():
    return 'Welcome!'


@sio.on('connect')
def connect(sid, environ):
    print('connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }
             )

if __name__ == '__main__':
    #app.run(port=2000)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


