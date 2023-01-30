FROM python:3.8
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /flaskapi-app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY ./gaze ./gaze
CMD ["python", "./main.py"]
