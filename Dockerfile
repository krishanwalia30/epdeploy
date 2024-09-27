# this sets up the container with python 3.10
FROM python:3.10-slim

# this copies everything in the Dockerfile directory to /app directory in tyhe container
COPY . /app

# change working dir
WORKDIR /app

# Adding Git to the image
# RUN apt-get update && apt-get install -y git
# RUN apt-get update && apt-get install libgl1
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# installing the dependencies
RUN pip install -r requirements.txt

# exposing port 80
EXPOSE 80

# command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]