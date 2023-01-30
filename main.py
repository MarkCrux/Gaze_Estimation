import cv2
from gaze.gaze_estimate import GazeEstimate
from flask import Flask, send_file, Response

class vc(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        gaze = GazeEstimate()
        success, image = self.cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        gaze.process(image)
        pose_direction = gaze.pose_direction()
        eye_ratio = gaze.eye_direction()

        image = gaze.annotated_frame(pose_direction, eye_ratio)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # success, image = self.cap.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


app = Flask(__name__)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')

def video_feed():
    return Response(gen(vc()), mimetype = 'multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run()