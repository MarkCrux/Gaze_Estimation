FROM python:3.8
WORKDIR /flaskapi-app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY ./gaze .
CMD ["python", "./main.py"]
