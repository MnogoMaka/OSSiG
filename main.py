import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import requests
import numpy as np
import cv2
from sympy.physics.units import moles
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
app = FastAPI()
# Загрузка моделей
models = {
    "sootv": YOLO("models/z1/sootv.pt"),
    "cld": YOLO('models/z2/clear_dirt.pt'),
    "det_car": YOLO('models/z3/detect_car_fill.pt'),
    "cls_grz": YOLO('models/z3/class_fkko.pt'),
    "det_front": YOLO('models/z4/detect_car_front.pt')
}
# Определяем структуру запросов
class ImageRequest(BaseModel):
    photo_url: str
    timestamp: datetime
    status: str
    trip_time: datetime
    trip_number: str
class FraudImageRequest(BaseModel):
    photo_list: str
    timestamp: datetime
    status: str
    trip_time: datetime
    trip_number: str
# Определяем структуру ответа
class ImageResponse(BaseModel):
    command_id: str
    response_status: str
    results: dict
# Функция для загрузки изображения по URL
def load_image(photo_url: str):
    response = requests.get(photo_url)
    if response.status_code == 200:
        image = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=400, detail="Ошибка загрузки изображения")
# Функция для обработки задачи
@app.post("/task/accordance", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    photo_url = request.get("photo_url")
    command_id = str(uuid.uuid4())
    if not photo_url:
        raise HTTPException(status_code=400, detail="Необходимые параметры отсутствуют")
    try:
        response = requests.get(photo_url)
        response.raise_for_status()
        image_c = Image.open(BytesIO(response.content))
        results = models["sootv"].predict(image_c)
        probs = results[0].probs
        pred_class_id = probs.top1
        confidence = probs.top1conf.item()
        label = results[0].names[int(pred_class_id)]
        results =  {"answer": label, "confidence": confidence}

        response = {
            "command_id": command_id,
            "response_status": "success",
            "results": results
        }
        return response
    except Exception as e:
        response = {
            "command_id": command_id,
            "response_status": 'Error',
            "results": {"error": str(e)}
        }
        return response


@app.post("/task/cargo", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    photo_url = request.get("photo_url")
    command_id = str(uuid.uuid4())
    if not photo_url:
        raise HTTPException(status_code=400, detail="Необходимые параметры отсутствуют")
    try:
        response = requests.get(photo_url)
        response.raise_for_status()
        image_c = Image.open(BytesIO(response.content))
        results = models["cld"].predict(image_c)
        probs = results[0].probs
        pred_class_id = probs.top1
        confidence = probs.top1conf.item()
        label = results[0].names[int(pred_class_id)]
        results = {"answer": label, "confidence": confidence}

        response = {
            "command_id": command_id,
            "response_status": "success",
            "results": results
        }
        return response
    except Exception as e:
        response = {
            "command_id": command_id,
            "response_status": 'Error',
            "results": {"error": str(e)}
        }
        return response

@app.post("/task/pollution", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    photo_url = request.get("photo_url")
    command_id = str(uuid.uuid4())
    if not photo_url:
        raise HTTPException(status_code=400, detail="Необходимые параметры отсутствуют")
    try:
        response = requests.get(photo_url)
        response.raise_for_status()
        image_c = Image.open(BytesIO(response.content))
        results = models['det_car'](image_c)
        if len(results[0].boxes) > 0:
            img = cv2.imread(image_c)
            best_box = max(results[0].boxes, key=lambda box: box.conf.item())
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            results = models['cls_grz'].predict(cropped_img)
            probs = results[0].probs
            pred_class_id = probs.top1
            confidence = probs.top1conf.item()
            label = results[0].names[int(pred_class_id)]
            results = {"answer": label, "confidence": confidence}

            response = {
                "command_id": command_id,
                "response_status": "success",
                "results": results
            }
            return response
        else:
            response = {
                "command_id": command_id,
                "response_status": 'Error',
                "results": {"error": 'The car was not detected'}
            }
            return response
    except Exception as e:
        response = {
            "command_id": command_id,
            "response_status": 'Error',
            "results": {"error": str(e)}
        }
        return response
# Эндпоинт для обработки изображения
@app.post("/task/fraud", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    photos_url = request.get("photo_list")
    command_id = str(uuid.uuid4())
    if not photos_url:
        raise HTTPException(status_code=400, detail="Необходимые параметры отсутствуют")
    try:
        for photos in photos_url:
            photos_in = photos.get('photo_numberplate_IN')
            photos_out = photos.get('photo_numberplate_OUT')
            response = requests.get(photos)
            response.raise_for_status()
    except Exception as e:
        response = {
            "command_id": command_id,
            "response_status": 'Error',
            "results": {"error": str(e)}
        }
        return response
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)