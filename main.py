from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import requests as req
from io import BytesIO
import numpy as np
import cv2
import base64
import random
import os
from pathlib import Path
import urllib.request as urlreq


class ModelInput(BaseModel):
    image_url: str = ""
    index: int = 0

app = FastAPI()
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_methods=["*"],
   allow_headers=["*"],
)


model = YOLO('yolov8n-seg.pt')

def detected_objects(image_path):
    # img_data = req.get(image_url).content
    print("FUNC Called with image url")
    # img = Image.open(BytesIO(img_data))
    # img.save("")

    # Actual CODE
    # results = model.predict(image_url)
    print(image_path)
    results = model.predict(source=image_path, retina_masks=True)
    # results = model.predict(source="content/cup.png")

    categories = results[0].names

    dc = []
    for i in range(len(results[0])):
        cat = results[0].boxes[i].cls
        dc.append(categories[int(cat)])

    print(dc)
    return results, dc
    # return 0,0

@app.post('/detected-objects-from-url')
def detect_objects_from_url(input_parameters: ModelInput):
    """
        Request: Pass the image file to be processed. The file should be sent as form-data with the key 'image'

        Response: Returns a JSON with the detected objects and the base64 encoded image.
    """
    # Accessing the request parameters
    image_url = input_parameters.image_url
    user_id = random.randint(0,10000000)

    # Creating the directory to save the user query
    os.makedirs("userQueries/", exist_ok=True)
    user_query_image_path = f"userQueries/user_{user_id}.png"

    # Image Downloading Headers
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    request = urlreq.Request(image_url, None, headers={"User-Agent": user_agent})

    # Making the request and downloading the image
    response = urlreq.urlopen(request)
    image_data_to_save = response.read()

    # Saving the image to the user query directory
    with open(user_query_image_path, 'wb') as file:
        file.write(image_data_to_save)
    
    print("Image has been received from the URL ")

    # Detecting objects in the image
    results, dc = detected_objects(user_query_image_path)

    # Deleting the image from the user query directory
    os.remove(user_query_image_path)
    
    # Returning the response
    return {"Objects_Detected": dc}


@app.post("/extracted-object-from-image-url")
def extracted_image(input_parameters: ModelInput):
    """
        Request: Pass the index of the object to be extracted from the image already passed to the api endpoint '/detected-objects'

        Response: Returns a JSON with the base64 encoded image of the extracted object.
    """
    # Accessing the request parameters
    index_of_the_choosen_detected_object = input_parameters.index
    image_url = input_parameters.image_url
    user_id = random.randint(0,10000000)

    # Creating the directory to save the user query
    os.makedirs("userQueries/", exist_ok=True)
    user_query_image_path = f"userQueries/user_{user_id}.png"

    # Image Downloading Headers
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    request = urlreq.Request(image_url, None, headers={"User-Agent": user_agent})

    # Making the request and downloading the image
    response = urlreq.urlopen(request)
    image_data_to_save = response.read()

    # Saving the image to the user query directory
    with open(user_query_image_path, 'wb') as file:
        file.write(image_data_to_save)
    
    print("Image has been received from the URL ")

    print(index_of_the_choosen_detected_object)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    results, dc = detected_objects(user_query_image_path)

    for result in results:
        # boxes = result.boxes
        img = np.copy(result.orig_img)

        c = result[index_of_the_choosen_detected_object]
        # print("The Valus is : ",c)
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.ones(img.shape[:2], np.uint8)

        # Create contour mask 
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Isolate object with transparent background (when saved as PNG)
        isolated = np.dstack([img, b_mask])

        # TODO: Actions here
        # Convert to RGB for PIL
        thatz = cv2.cvtColor(isolated, cv2.COLOR_BGR2RGBA)

        buffered = BytesIO()

        to_save = Image.fromarray(thatz)
        # to_save_after_bg_removal = remove(to_save)

        # to_save.save(buffered, format='PNG')
        to_save.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue())

        os.makedirs("content/", exist_ok=True)
        os.remove(user_query_image_path)

        to_save.save(f"content/{user_id}_{label}.png")

        os.remove(f"content/{user_id}_{label}.png")

        return {
            "Object Extraction": "Successful",
            "image_str": img_str
        }

    # bbox=boxes.xyxy.tolist()[index_of_the_choosen_detected_object]
    # for result in resul
