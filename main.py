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

@app.post("/task/pollution", response_model=ImageResponse)
async def process_pollution(request: ImageRequest):

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

@app.post("/task/cargo", response_model=ImageResponse)
async def process_cargo(request: ImageRequest):

    try:
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

                # Заменяем NaN на None и фильтруем данные
                df = df.where(pd.notnull(df), None)
                #print(1, df)
                #print(2, df_filtered)
                # Обрабатываем каждую запись
                for _, row in df.iterrows():
                    filename = row.get('filename2', '')
                    number = filename.split(',')[0] if filename else None
                    #print(number)
                    fraud_descr = row.get('fraud_descr')
                    # Уточняем проверку для строк с пробелами
                    value = 0 if pd.isna(fraud_descr) or str(fraud_descr).strip() == '' else fraud_descr

                    if number:
                        d = {'trip_number': number, 'status': value}
                        results.append(d)

        except Exception as e:
            raise ValueError(f"Ошибка обработки архива: {str(e)}") from e

    return results


@app.post("/task/fraud", response_model=FraudImageResponse)
async def process_fraud(request: FraudImageRequest):
    temp_dir = TEMP_DIR / uuid.uuid4().hex
    photos_dir_all = temp_dir / "any"  # Основная папка для фотографий
    photos_dir_one = temp_dir / "main_trip"
    photos_dir_all.mkdir(parents=True, exist_ok=True)
    photos_dir_one.mkdir(parents=True, exist_ok=True)
    zip_path = None
    result_zip = None

    try:
        # Скачиваем и сохраняем изображения
        #print(request.photo_list)
        names = [d.trip_number for d in request.photo_list]
        if request.trip_check not in names:
            raise HTTPException(
                status_code=400,
                detail=f"Отсутствует искомый рейс {request.trip_check}"
            )

        for photo_data in request.photo_list:
            #print(photo_data)
            for photo_type in ['photo_numberplate_IN', 'photo_numberplate_OUT']:
                if not hasattr(photo_data, photo_type):
                    #print(f"Attribute '{photo_type}' not found in {photo_data}")  # Сообщение об отсутствии атрибута
                    continue
                # Скачиваем изображение по URL
                try:
                    value = getattr(photo_data, photo_type)
                    response = requests.get(value, timeout=10, verify=False)
                    response.raise_for_status()
                    image_data = response.content
                    # Формируем имя файла
                    filename = (
                        f"{getattr(photo_data, 'trip_number')}_"
                        f"{getattr(photo_data, 'car_number')}_"
                        f"{getattr(photo_data, 'trip_datetime').replace(':', '_')}_"
                        f"{'_'.join(photo_type.split('_')[1:])}.jpg"
                    )
                    # Сохраняем в подпапку
                    if request.trip_check == photo_data.trip_number:
                        if 'OUT' in filename:
                            continue
                        with open(photos_dir_one / filename, 'wb') as file:
                            file.write(image_data)
                    else:
                        with open(photos_dir_all / filename, 'wb') as file:
                            file.write(image_data)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Ошибка загрузки {photo_type}: {str(e)}"
                    )

        # Создаем ZIP-архивы с сохранением структуры папок
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

                "file": (zip_path2.name, f_main, 'application/zip'),  # Основной архив
                "relations_file": (zip_path1.name, f_relations, 'application/zip')  # Архив для сравнения
            }

            data = {
                "strategy_type": "in-in (m)",
                "threshold": "30"
            }

            api_response = requests.post(
                API_URL,
                files=files,
                data=data,
                timeout=8000  #
            )
            api_response.raise_for_status()

            print("Запрос успешен. Ответ:", api_response.status_code)
            if "application/zip" not in api_response.headers.get("Content-Type", ""):
                raise Exception(f"API вернул ошибку: {api_response.text}")

        # Обработка результата
        result_zip = TEMP_DIR / f"result_{uuid.uuid4().hex}.zip"
        result_zip.write_bytes(api_response.content)
        #print(getattr(request, 'trip_check'))
        fraud_results = process_zip_archive(result_zip, trip_check=getattr(request, 'trip_check'))
        fraud_status = False if set([i.get('status') for i in fraud_results]) == {0} else True

        print(FraudImageResponse(
            response_status="success",
            trip_check=request.trip_check,
            results={
                "fraud_status": fraud_status,
                "results": fraud_results
            }
        ))
        return FraudImageResponse(
            response_status="success",
            trip_check=request.trip_check,
            results={
                "fraud_status": fraud_status,
                "results": fraud_results
            }
        )

    except Exception as e:
        return FraudImageResponse(
            response_status="error",
            trip_check=request.trip_check,
            results={"error": str(e)}
        )
    finally:
        # Очистка ресурсов
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if result_zip and result_zip.exists():
                result_zip.unlink()
        except Exception as cleanup_error:
            print(f"Ошибка очистки: {cleanup_error}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
