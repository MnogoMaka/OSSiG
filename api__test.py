import requests
from datetime import datetime, UTC
import json

BASE = "http://localhost:8000"

data = {
    "photo_url": "https://ossig.smart.mos.ru/backup/kpts3/24d1ec6f-fe81-457c-be91-7c3e22d2a682/snapshot.jpg",
    "timestamp": datetime.utcnow().isoformat(),
    "status": "Pending",
    "trip_time": datetime.utcnow().isoformat(),
    "trip_number": "12345"
}

def test_accordance():
    res = requests.post(f"{BASE}/task/accordance", json=data)
    print("ACCORDANCE:", res.status_code, res.json())

def test_cargo():
    res = requests.post(f"{BASE}/task/cargo", json=data)
    print("CARGO:", res.status_code, res.json())

def test_pollution():
    res = requests.post(f"{BASE}/task/pollution", json=data)
    print("POLLUTION:", res.status_code, res.json())


def test_fraud():
    # Тестовые данные с временными метками в UTC
    fraud_data = {
        "photo_list": [
            {
                "trip_number": "111111111111",
                "trip_datetime": '2023_07_18_19_24_25',
                "car_number": "A123BC123797",
                "photo_numberplate_IN": "https://ossig.smart.mos.ru/backup/kpts3/2376b233-193a-45d6-b068-5ffd873e6602/numberplate.jpg",
                "photo_numberplate_OUT": "https://ossig.smart.mos.ru/backup/kpts3/de507c21-1029-4a94-8f0b-6058a65f31ad/numberplate.jpg"
            },
            {
                "trip_number": "222222222222",
                "trip_datetime": '2023_07_18_19_24_30',
                "car_number": "A123BC123797",
                "photo_numberplate_IN": "https://ossig.smart.mos.ru/backup/kpts3/89180804-8040-425c-81db-4f44d21302a8/numberplate.jpg",
                "photo_numberplate_OUT": "https://ossig.smart.mos.ru/backup/kpts3/fc05caa9-34a2-48a9-a8d3-2742ebd7c555/numberplate.jpg"
            }
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "test"
    }

    try:
        res = requests.post(f"{BASE}/task/fraud", json=fraud_data, timeout=8000)
        res.raise_for_status()

        response_data = res.json()
        print("\nFraud Test Results:")
        print(f"Status Code: {res.status_code}")
        print("Response Body:")
        print(json.dumps(response_data, indent=2))

        return response_data

    except requests.RequestException as e:
        print(f"\nFraud Test Failed: {str(e)}")
        if e.response is not None:
            print(f"Error Response: {e.response.text}")
        return None

if __name__ == "__main__":
    test_accordance()
    test_cargo()
    test_pollution()
    test_fraud()