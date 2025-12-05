import os
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)


def get_core_api_instance():
    """Создает экземпляр CorePipeline из конфига API"""
    # Определяем корень core директории
    current_dir = os.path.dirname(os.path.abspath(__file__))
    core_root = os.path.dirname(current_dir)  # Поднимаемся на уровень выше из utils/
    config_dir = os.path.join(core_root, "config")
    
    # Очищаем предыдущие инициализации Hydra
    GlobalHydra.instance().clear()
    
    # Инициализируем Hydra с указанием абсолютного пути к конфигу
    with initialize_config_dir(config_dir=config_dir, version_base="1.2.0"):
        cfg = compose(config_name="config_api")
        return instantiate(cfg["core"])
