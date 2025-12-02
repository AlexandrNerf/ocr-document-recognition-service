import asyncio
import uuid
from typing import Optional

import rootutils
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)

from data.data_classes import Prediction
from pipelines.run import CorePipeline
from utils.get_core import get_shift_ocr_instance
from utils.base64utils import decode_image
from utils.BBoxVisualizer import BoundingBoxVisualizer
from utils.html_generator import fig_to_html

app = FastAPI()

# Хранилище задач (в продакшене лучше использовать Redis или БД)
tasks: dict[str, dict] = {}

# Создаем экземпляр пайплайна (можно переиспользовать)
_pipeline_instance: Optional[CorePipeline] = None


def get_pipeline() -> CorePipeline:
    """Получает или создает экземпляр пайплайна"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = get_shift_ocr_instance()
    return _pipeline_instance


def process_image_sync(task_id: str, image_data: bytes, file_format: str):
    """Синхронная обработка изображения (будет запущена в отдельном потоке)"""
    try:
        tasks[task_id]["status"] = "processing"
        
        # Декодируем изображение
        import base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image = decode_image((image_base64, file_format))
        
        # Получаем пайплайн и обрабатываем
        pipeline = get_pipeline()
        result = pipeline.predict(image)
        
        # Генерируем HTML из plotly figure
        if result["images"] and result["predictions"]:
            image_array = result["images"][0]
            predictions = result["predictions"][0]
            
            # Создаем plotly figure
            fig = BoundingBoxVisualizer.show_image(
                image_array, 
                predictions, 
                max_width=1000
            )
            
            # Конвертируем в HTML
            html_content = fig_to_html(fig)
            
            tasks[task_id]["status"] = "done"
            tasks[task_id]["html"] = html_content
            tasks[task_id]["predictions"] = [
                {
                    "text": pred.text,
                    "score": float(pred.score) if pred.score is not None else None,
                    "text_score": float(pred.text_score) if pred.text_score is not None else None,
                    "absolute_box": [[int(x), int(y)] for x, y in pred.absolute_box],
                }
                for pred in predictions
            ]
        else:
            tasks[task_id]["status"] = "done"
            tasks[task_id]["html"] = "<html><body><p>No detections found</p></body></html>"
            tasks[task_id]["predictions"] = []
            
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    file_format: str = Form("image")
):
    """
    POST эндпоинт для загрузки и обработки файла.
    
    Args:
        file: Загружаемый файл (изображение или PDF)
        file_format: Формат файла ("image" или "pdf")
        
    Returns:
        JSON с task_id
    """
    # Генерируем уникальный ID задачи
    task_id = str(uuid.uuid4())
    
    # Читаем файл
    try:
        file_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Инициализируем задачу
    tasks[task_id] = {
        "status": "pending",
        "html": None,
        "predictions": None,
        "error": None,
    }
    
    # Запускаем асинхронную обработку в отдельном потоке
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, process_image_sync, task_id, file_data, file_format)
    
    return JSONResponse(content={"task_id": task_id})


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    GET эндпоинт для получения статуса обработки.
    
    Args:
        task_id: ID задачи
        
    Returns:
        - Если обрабатывается: {"status": "processing"}
        - Если готово: {"status": "done", "html": "..."}
        - Если ошибка: {"status": "error", "error": "..."}
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    status = task["status"]
    
    if status == "processing" or status == "pending":
        return JSONResponse(content={"status": "process"})
    elif status == "done":
        return JSONResponse(
            content={
                "status": "done",
                "html": task["html"],
                "predictions": task["predictions"],
            }
        )
    elif status == "error":
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": task.get("error", "Unknown error"),
            }
        )
    else:
        return JSONResponse(content={"status": status})


@app.get("/result/{task_id}", response_class=HTMLResponse)
async def get_result_html(task_id: str):
    """
    GET эндпоинт для получения HTML результата.
    
    Args:
        task_id: ID задачи
        
    Returns:
        HTML страница с результатами
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    status = task["status"]
    
    if status == "processing" or status == "pending":
        return HTMLResponse(content="<html><body><p>Processing...</p></body></html>")
    elif status == "done":
        return HTMLResponse(content=task["html"])
    elif status == "error":
        error_msg = task.get("error", "Unknown error")
        return HTMLResponse(
            content=f"<html><body><p>Error: {error_msg}</p></body></html>",
            status_code=500
        )
    else:
        return HTMLResponse(content=f"<html><body><p>Status: {status}</p></body></html>")


PORT = 8000


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
