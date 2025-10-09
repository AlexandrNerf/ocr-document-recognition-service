import rootutils
from hydra import compose, initialize
from hydra.utils import instantiate

rootutils.setup_root(__file__, indicator=".project-roots", pythonpath=True)


def get_shift_ocr_instance():
    with initialize(config_path="../../config", version_base="1.2.0"):
        config = compose(config_name="config")

    return instantiate(config.shift_ocr)


if __name__ == "__main__":
    print(get_shift_ocr_instance())
