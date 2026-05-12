# Skin Type Classifier — ML Service

FastAPI микросервис для определения типа кожи по фотографии лица.

## Классы
- `combination` — комбинированная
- `dry` — сухая
- `normal` — нормальная
- `oily` — жирная

## Метрики модели (EfficientNet-B2)

| Класс | Precision | Recall | F1 |
|---|---|---|---|
| combination | 0.96 | 0.97 | 0.96 |
| dry | 0.91 | 0.93 | 0.92 |
| normal | 0.88 | 0.91 | 0.89 |
| oily | 0.93 | 0.89 | 0.91 |
| **overall** | **0.92** | **0.92** | **0.92** |

## Структура

```
ml-service/
├── src/
│   └── app.py          # FastAPI приложение
├── models/
│   └── efficientnet_merged.pth   # модель (не в git — скачать отдельно)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Запуск локально

```bash
pip install -r requirements.txt
MODEL_PATH=models/efficientnet_merged.pth API_KEY=your-secret uvicorn src.app:app --reload
```

## Запуск через Docker

```bash
docker build -t skin-classifier .
docker run -p 8000:8000 \
  -e API_KEY=your-secret \
  -v $(pwd)/models:/app/models \
  skin-classifier
```

## API

### GET /health
```json
{"status": "ok", "model_version": "1.0.0", "device": "cpu"}
```

### POST /predict
Заголовок: `x-api-key: your-secret`
Тело: `multipart/form-data`, поле `file` — изображение (jpg/png/webp)

**Успешный ответ:**
```json
{
  "model_version": "1.0.0",
  "skin_type": "oily",
  "confidence": 0.91,
  "probabilities": {
    "combination": 0.02,
    "dry": 0.04,
    "normal": 0.03,
    "oily": 0.91
  },
  "warnings": []
}
```

**Пример запроса из Go:**
```go
req, _ := http.NewRequest("POST", "http://ml-service:8000/predict", body)
req.Header.Set("x-api-key", os.Getenv("ML_API_KEY"))
req.Header.Set("Content-Type", writer.FormDataContentType())
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODEL_PATH` | `models/efficientnet_merged.pth` | Путь к весам модели |
| `API_KEY` | `changeme-secret-key` | Ключ для защиты эндпоинта |
| `MODEL_VERSION` | `1.0.0` | Версия модели в ответе |
