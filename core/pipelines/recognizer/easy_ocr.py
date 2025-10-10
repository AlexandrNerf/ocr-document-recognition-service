from typing import Optional

import easyocr
import torch
from data.data_classes import Prediction
from pipelines.default.recognizer import Recognizer


class EasyOCRRecognizer(Recognizer):
    def _init(
        self,
        cuda: bool,
        model_storage_directory: str,
        user_network_directory: str,
        verbose: bool,
        quantize: bool,
        download_enabled: bool,
        lang_list: Optional[list] = ["en", "ru", "tjk"],
    ):
        """
        Recognizer из EasyOCR

        Args:
            lang_list (list): Словари языков для распознавания
            cuda (bool): Использовать GPU
            model_storage_directory (str): Путь до модели.
                    Если пустой, выбирает из переменных среды
            user_network_directory (str): Путь до пользовательской модели
            verbose (bool): Выводить комментарии
            quantize (bool): Квантизация сети
            download_enabled (bool): Включить загрузку модели из сети
        """
        self.reader = easyocr.Reader(
            lang_list=lang_list,
            gpu=cuda and torch.cuda.is_available(),
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory,
            verbose=verbose,
            quantize=quantize,
            download_enabled=download_enabled,
            detector=False,
        )

    def recognize(self, post_detections) -> list[Prediction]:
        recognitions: list[Prediction] = []

        for det in post_detections:
            roi = det.crop
            if roi.size == 0:
                det.text = ""
                det.text_score = 1.0
                continue

            results = self.reader.recognize(roi)

            if results:
                text: str = results[0][1]
                score: float = results[0][2]
            else:
                text, score = "", 1.0

            det.text = text
            det.text_score = score

            recognitions.append(det)

        return recognitions
