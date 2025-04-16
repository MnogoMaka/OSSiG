from dotenv import load_dotenv
import uuid
import tempfile
from typing import Dict, List
from datetime import datetime
from io import BytesIO
from pathlib import Path
from fastapi.encoders import jsonable_encoder
import cv2
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, HttpUrl
from PIL import Image
import zipfile
import shutil
from ultralytics import YOLO
import os
from pathlib import Path

load_dotenv()
app = FastAPI()

API_URL = os.getenv("API_URL")
if not API_URL:
    raise RuntimeError("API_URL не задан в .env файле")
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
MAX_IMAGE_SIZE = (1024, 1024)

MODELS = {
    "sootv": YOLO("models/z1/sootv.pt"),
    "cld": YOLO('models/z2/clear_dirt.pt'),
    "det_car": YOLO('models/z3/detect_car_fill.pt'),
    "cls_grz": YOLO('models/z3/class_fkko.pt'),
    "det_front": YOLO('models/z4/detect_car_front.pt')
}


class ImageRequest(BaseModel):
    photo_url: HttpUrl
    timestamp: datetime
    status: str
    trip_time: datetime
    trip_number: str


class FraudImageRequest(BaseModel):
    photo_list: list
    timestamp: datetime
    status: str


class ImageResponse(BaseModel):
    trip_number: str
    response_status: str
    results: dict


class FraudImageResponse(BaseModel):
    response_status: str
    results: dict


def download_image(url: HttpUrl) -> Image.Image:

    response = requests.get(url, timeout=80)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    image.thumbnail(MAX_IMAGE_SIZE)
    return image.convert("RGB")


def process_detection(model_key: str, image: Image.Image) -> dict:

    model = MODELS[model_key]
    results = model.predict(image)

    if not results:
        raise ValueError("No results from model prediction")

    probs = results[0].probs
    return {
        "answer": results[0].names[probs.top1],
        "confidence": round(probs.top1conf.item(), 4)
    }


@app.post("/task/accordance", response_model=ImageResponse)
async def process_accordance(request: ImageRequest):

    try:
        image = download_image(request.photo_url)
        prediction = process_detection("sootv", image)

        return ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )
    except Exception as e:
        return ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

@app.post("/task/cargo", response_model=ImageResponse)
async def process_cargo(request: ImageRequest):

    try:
        image = download_image(request.photo_url)
        prediction = process_detection("cld", image)

        return ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )
    except Exception as e:
        return ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

@app.post("/task/pollution", response_model=ImageResponse)
async def process_pollution(request: ImageRequest):

    try:
        image = download_image(request.photo_url)
        detection_results = MODELS["det_car"].predict(image)

        if not detection_results[0].boxes:
            raise ValueError("No vehicles detected in the image")

        best_box = max(detection_results[0].boxes, key=lambda b: b.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cropped = cv_image[y1:y2, x1:x2]

        # Convert back to PIL Image for classification
        classification_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        prediction = process_detection("cls_grz", classification_image)

        return ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )
    except Exception as e:
        return ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

def process_zip_archive(zip_path: Path) -> List[Dict]:

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if "report.xlsx" not in zip_ref.namelist():
                    raise FileNotFoundError("Файл отчета не найден")

                zip_ref.extract("report.xlsx", tmp_dir)
                df = pd.read_excel(
                    Path(tmp_dir) / "report.xlsx",
                    engine='openpyxl',
                    dtype={"race_id": str, "fraud_descr": str},
                )

                df = df.where(pd.notnull(df), None)

                # Формируем результат
                results = [
                    {
                        "trip_number": str(row["race_id"]),
                        "status": row["fraud_descr"] if row["fraud_descr"] else None
                    }
                    for _, row in df.iterrows()
                ]

        except Exception as e:
            raise ValueError(f"Ошибка обработки архива: {str(e)}") from e

    return results

def normalize_status(status: str):
    if status is None:
        return None

    # Список ожидаемых статусов
    expected_statuses = {
        "possible_fraud": "Возможно мошенничество",
        "different_cars": "Машины различаются"
    }

    # Проверка на прямой текст
    if status in expected_statuses.values():
        return status

    # Проверка на английские варианты
    if status in expected_statuses:
        return expected_statuses[status]

    # Декодирование Unicode escape
    try:
        decoded = status.encode('latin-1').decode('unicode-escape')
        if decoded in expected_statuses.values():
            return decoded
    except:
        pass

    return status

@app.post("/task/fraud", response_model=FraudImageResponse)
async def process_fraud(request: FraudImageRequest):
    temp_dir = TEMP_DIR / uuid.uuid4().hex
    photos_dir = temp_dir / "photos"  # Основная папка для фотографий
    photos_dir.mkdir(parents=True, exist_ok=True)
    zip_path = None
    result_zip = None

    try:
        # Скачиваем и сохраняем изображения
        for photo_data in request.photo_list:
            for photo_type in ['photo_numberplate_IN', 'photo_numberplate_OUT']:
                if photo_type not in photo_data:
                    continue

                # Скачиваем изображение по URL
                try:
                    response = requests.get(photo_data[photo_type], timeout=80)
                    response.raise_for_status()
                    image_data = response.content
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Ошибка загрузки {photo_type}: {str(e)}"
                    )

                # Формируем имя файла
                filename = (
                    f"{photo_data['trip_number']}_"
                    f"{photo_data['car_number']}_"
                    f"{photo_data['trip_datetime'].replace(':', '_')}_"
                    f"{"_".join(photo_type.split('_')[1:])}.jpg"
                )

                # Сохраняем в подпапку photos
                (photos_dir / filename).write_bytes(image_data)

        # Создаем ZIP-архив с сохранением структуры папок
        zip_path = temp_dir.with_suffix('.zip')
        shutil.make_archive(
            base_name=str(temp_dir),
            format='zip',
            root_dir=str(temp_dir),
            base_dir='photos'
        )

        # Отправляем архив
        with zip_path.open('rb') as f:
            api_response = requests.post(
                API_URL,
                files={"file": (f"photos_{zip_path.name}", f)},
                timeout=8000
            )
            api_response.raise_for_status()

        # Обработка результата
        result_zip = TEMP_DIR / f"result_{uuid.uuid4().hex}.zip"
        result_zip.write_bytes(api_response.content)

        fraud_results = process_zip_archive(result_zip)
        fraud_status = any(
            "Мошенничество" in str(res["status"]) or
            "Машины различаются" in str(res["status"])
            for res in fraud_results
        )
        for res in fraud_results:
            res["status"] = normalize_status(res["status"])
        print(FraudImageResponse(
            response_status="success",
            results={
                "query_status": fraud_status,
                "results": fraud_results
            }
        ))
        return FraudImageResponse(
            response_status="success",
            results={
                "query_status": fraud_status,
                "results": fraud_results
            }
        )

    except Exception as e:
        return FraudImageResponse(
            response_status="error",
            results={"error": str(e)}
        )
    finally:
        # Очистка ресурсов
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if zip_path and zip_path.exists():
                zip_path.unlink()
            if result_zip and result_zip.exists():
                result_zip.unlink()
        except Exception as cleanup_error:
            print(f"Ошибка очистки: {cleanup_error}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)