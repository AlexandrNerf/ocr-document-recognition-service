import rootutils
import uvicorn

rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)
from data.data_classes import Prediction
from fastapi import FastAPI
from pipelines.default.model import ImageBase64
from pipelines.run import CorePipeline

from core.utils.get_core import get_shift_ocr_instance

app = FastAPI()
core_instance: CorePipeline = get_shift_ocr_instance()

PORT = 8000


@app.post("/predictText")
def predict_text(img_bytes: ImageBase64, file_format: str) -> list[Prediction]:
    """
    Предсказание текста на изображении.
    Параметры:
    - **img_bytes**: Файл в формате Base64, передаваемый в теле запроса.
    - **file_format**: Формат файла (image, pdf).
    Возвращает:
    - Список объектов Detection с предсказанным текстом и уверенностью.
    """
    img = core_instance._loader.load(base64=(img_bytes.data, file_format))
    pred: list[Prediction] = core_instance.predict(img)
    return pred


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
