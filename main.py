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
    parameters: dict
    timestamp: datetime
    status: str
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
def process_task(task_type: str, image, parameters: dict):
    if task_type == "accordance" :
        try:
            response = requests.get(image)
            response.raise_for_status()
            image_c = Image.open(BytesIO(response.content))
            results = models["sootv"].predict(image_c)
            probs = results[0].probs
            pred_class_id = probs.top1
            confidence = probs.top1conf.item()
            label = results[0].names[int(pred_class_id)]
        except Exception as e:
            return {"error": str(e)}
        return {"answer": label, "confidence": confidence}
    elif task_type == "cargo":
        try:
            response = requests.get(image)
            response.raise_for_status()
            image_c = Image.open(BytesIO(response.content))
            results = models["cld"].predict(image_c)
            probs = results[0].probs
            pred_class_id = probs.top1
            confidence = probs.top1conf.item()
            label = results[0].names[int(pred_class_id)]
        except Exception as e:
            return {"error": str(e)}
        return {"answer": label, "confidence": confidence}
    elif task_type == "pollution":
        pass
    elif task_type == "fraud":
        pass
    else:
        raise HTTPException(status_code=400, detail="Неверный тип задачи")
# Эндпоинт для обработки изображения
@app.post("/process_image", response_model=ImageResponse)
async def process_image(request: ImageRequest):
    parameters = request.parameters
    task_type = parameters.get("task_type")
    photo_url = parameters.get("photo_url")
    if not task_type or not photo_url:
        raise HTTPException(status_code=400, detail="Необходимые параметры отсутствуют")
    # Обрабатываем задачу
    results = process_task(task_type, photo_url, parameters)
    # Создаем ответ
    command_id = str(uuid.uuid4())
    response = {
        "command_id": command_id,
        "response_status": "success",
        "results": results
    }
    return response
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)