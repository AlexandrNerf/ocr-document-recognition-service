import logging
import time
from abc import ABC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class BasePipeline(ABC):
    def __init__(self, *args, **kwargs):
        logging.info(f'Инициализация пайплайна {self.__class__.__name__} начата')
        self._pipeline_storage = {}
        try:
            self._init(*args, **kwargs)
            logging.info(f'Пайплайн {self.__class__.__name__} инициализирован')
        except Exception as e:
            logging.exception(f'Ошибка при инициализации пайплайна {self.__class__.__name__}: {e}')

    def _init(self, *args, **kwargs):
        pass

    def run(self, data):
        try:
            logging.info(f'Запущен пайплайн {self.__class__.__name__}')
            self._pipeline_storage = data
            self._pipeline_storage.update(self._run(self._pipeline_storage) or {})

            logging.info(f'Пайплайн {self.__class__.__name__} успешно отработал')
            return self._pipeline_storage
        except Exception as e:
            logging.exception(f'Ошибка при запуске пайплайна {self.__class__.__name__}: {e}')
            return

    def _run(self, data):
        pass