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
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import logging
from logging.handlers import RotatingFileHandler
import json


# Настройка логирования
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("api_logger")
    logger.setLevel(logging.INFO)

    # Форматирование
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(endpoint)s] - %(message)s'
    )

    # Файловый обработчик с ротацией
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Инициализация логгера
logger = setup_logging()


def log_request(endpoint: str, request_data: dict, additional_info: dict = None):
    """Логирование входящего запроса"""
    log_data = {
        "endpoint": endpoint,
        "request": request_data,
        "type": "request"
    }
    if additional_info:
        log_data.update(additional_info)

    logger.info(f"Request received", extra={"endpoint": endpoint})
    logger.debug(f"Request details: {json.dumps(log_data, default=str)}", extra={"endpoint": endpoint})


def log_response(endpoint: str, response_data: dict, processing_time: float = None):
    """Логирование исходящего ответа"""
    log_data = {
        "endpoint": endpoint,
        "response": response_data,
        "type": "response"
    }
    if processing_time is not None:
        log_data["processing_time_seconds"] = processing_time

    status = response_data.get('response_status', 'unknown')
    logger.info(f"Response sent - Status: {status}", extra={"endpoint": endpoint})
    logger.debug(f"Response details: {json.dumps(log_data, default=str)}", extra={"endpoint": endpoint})


def log_error(endpoint: str, error: str, details: dict = None):
    """Логирование ошибок"""
    error_data = {
        "endpoint": endpoint,
        "error": error,
        "type": "error"
    }
    if details:
        error_data.update(details)

    logger.error(f"Error occurred: {error}", extra={"endpoint": endpoint})
    logger.error(f"Error details: {json.dumps(error_data, default=str)}", extra={"endpoint": endpoint})


load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class FraudData(BaseModel):
    trip_number: str
    trip_datetime: str
    car_number: str
    photo_numberplate_IN: HttpUrl
    photo_numberplate_OUT: HttpUrl


class FraudImageRequest(BaseModel):
    trip_check: str
    timestamp: datetime
    status: str
    photo_list: List[FraudData]


class ImageResponse(BaseModel):
    trip_number: str
    response_status: str
    results: dict


class FraudImageResponse(BaseModel):
    response_status: str
    trip_check: str
    results: dict


def download_image(url: HttpUrl) -> Image.Image:
    response = requests.get(url, timeout=80, verify=False)
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
    start_time = datetime.now()
    endpoint = "/task/accordance"

    try:
        log_request(endpoint, request.dict(), {"trip_number": request.trip_number})

        image = download_image(request.photo_url)
        prediction = process_detection("sootv", image)

        response = ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, response.dict(), processing_time)

        return response

    except Exception as e:
        error_response = ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

        log_error(endpoint, str(e), {"trip_number": request.trip_number})
        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, error_response.dict(), processing_time)

        return error_response


@app.post("/task/pollution", response_model=ImageResponse)
async def process_pollution(request: ImageRequest):
    start_time = datetime.now()
    endpoint = "/task/pollution"

    try:
        log_request(endpoint, request.dict(), {"trip_number": request.trip_number})

        image = download_image(request.photo_url)
        prediction = process_detection("cld", image)

        response = ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, response.dict(), processing_time)

        return response

    except Exception as e:
        error_response = ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

        log_error(endpoint, str(e), {"trip_number": request.trip_number})
        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, error_response.dict(), processing_time)

        return error_response


@app.post("/task/cargo", response_model=ImageResponse)
async def process_cargo(request: ImageRequest):
    start_time = datetime.now()
    endpoint = "/task/cargo"

    try:
        log_request(endpoint, request.dict(), {"trip_number": request.trip_number})

        image = download_image(request.photo_url)
        detection_results = MODELS["det_car"].predict(image)

        if not detection_results[0].boxes:
            raise ValueError("No vehicles detected in the image")

        best_box = max(detection_results[0].boxes, key=lambda b: b.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cropped = cv_image[y1:y2, x1:x2]

        classification_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        prediction = process_detection("cls_grz", classification_image)

        response = ImageResponse(
            trip_number=request.trip_number,
            response_status="success",
            results=prediction
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, response.dict(), processing_time)

        return response

    except Exception as e:
        error_response = ImageResponse(
            trip_number=request.trip_number,
            response_status="error",
            results={"error": str(e)}
        )

        log_error(endpoint, str(e), {"trip_number": request.trip_number})
        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, error_response.dict(), processing_time)

        return error_response


def process_zip_archive(zip_path: Path, trip_check: str):
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

                for _, row in df.iterrows():
                    filename = row.get('filename2', '')
                    number = filename.split(',')[0] if filename else None
                    fraud_descr = row.get('fraud_descr')
                    value = 0 if pd.isna(fraud_descr) or str(fraud_descr).strip() == '' else fraud_descr

                    if number:
                        d = {'trip_number': number, 'status': value}
                        results.append(d)

        except Exception as e:
            raise ValueError(f"Ошибка обработки архива: {str(e)}") from e

    return results


@app.post("/task/fraud", response_model=FraudImageResponse)
async def process_fraud(request: FraudImageRequest):
    start_time = datetime.now()
    endpoint = "/task/fraud"

    try:
        log_request(endpoint, request.dict(), {
            "trip_check": request.trip_check,
            "photo_count": len(request.photo_list)
        })

        temp_dir = TEMP_DIR / uuid.uuid4().hex
        photos_dir_all = temp_dir / "any"
        photos_dir_one = temp_dir / "main_trip"
        photos_dir_all.mkdir(parents=True, exist_ok=True)
        photos_dir_one.mkdir(parents=True, exist_ok=True)
        zip_path = None
        result_zip = None

        names = [d.trip_number for d in request.photo_list]
        if request.trip_check not in names:
            error_msg = f"Отсутствует искомый рейс {request.trip_check}"
            raise HTTPException(status_code=400, detail=error_msg)

        for photo_data in request.photo_list:
            for photo_type in ['photo_numberplate_IN', 'photo_numberplate_OUT']:
                if not hasattr(photo_data, photo_type):
                    continue
                try:
                    value = getattr(photo_data, photo_type)
                    response = requests.get(value, timeout=10, verify=False)
                    response.raise_for_status()
                    image_data = response.content

                    filename = (
                        f"{getattr(photo_data, 'trip_number')}_"
                        f"{getattr(photo_data, 'car_number')}_"
                        f"{getattr(photo_data, 'trip_datetime').replace(':', '_')}_"
                        f"{'_'.join(photo_type.split('_')[1:])}.jpg"
                    )

                    if request.trip_check == photo_data.trip_number:
                        if 'OUT' in filename:
                            continue
                        with open(photos_dir_one / filename, 'wb') as file:
                            file.write(image_data)
                    else:
                        with open(photos_dir_all / filename, 'wb') as file:
                            file.write(image_data)

                except Exception as e:
                    error_msg = f"Ошибка загрузки {photo_type}: {str(e)}"
                    raise HTTPException(status_code=400, detail=error_msg)

        zip_path1 = photos_dir_one.with_suffix('.zip')
        zip_path2 = photos_dir_all.with_suffix('.zip')

        shutil.make_archive(
            base_name=str(photos_dir_one),
            format='zip',
            root_dir=str(temp_dir),
            base_dir='main_trip'
        )
        shutil.make_archive(
            base_name=str(photos_dir_all),
            format='zip',
            root_dir=str(temp_dir),
            base_dir='any'
        )

        for path in [zip_path1, zip_path2]:
            if not path.exists() or path.stat().st_size == 0:
                raise HTTPException(status_code=500, detail=f"Архив {path} не создан или пуст")

        with zip_path1.open('rb') as f_relations, zip_path2.open('rb') as f_main:
            files = {
                "file": (zip_path2.name, f_main, 'application/zip'),
                "relations_file": (zip_path1.name, f_relations, 'application/zip')
            }

            data = {
                "strategy_type": "in-in (m)",
                "threshold": "30"
            }

            api_response = requests.post(
                API_URL,
                files=files,
                data=data,
                timeout=8000
            )
            api_response.raise_for_status()

            logger.info(f"External API call successful: {api_response.status_code}", extra={"endpoint": endpoint})

            if "application/zip" not in api_response.headers.get("Content-Type", ""):
                raise Exception(f"API вернул ошибку: {api_response.text}")

        result_zip = TEMP_DIR / f"result_{uuid.uuid4().hex}.zip"
        result_zip.write_bytes(api_response.content)

        fraud_results = process_zip_archive(result_zip, trip_check=getattr(request, 'trip_check'))
        fraud_status = False if set([i.get('status') for i in fraud_results]) == {0} else True

        response = FraudImageResponse(
            response_status="success",
            trip_check=request.trip_check,
            results={
                "fraud_status": fraud_status,
                "results": fraud_results
            }
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, response.dict(), processing_time)

        return response

    except Exception as e:
        error_response = FraudImageResponse(
            response_status="error",
            trip_check=request.trip_check,
            results={"error": str(e)}
        )

        log_error(endpoint, str(e), {"trip_check": request.trip_check})
        processing_time = (datetime.now() - start_time).total_seconds()
        log_response(endpoint, error_response.dict(), processing_time)

        return error_response

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if result_zip and result_zip.exists():
                result_zip.unlink()
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}", extra={"endpoint": endpoint})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)