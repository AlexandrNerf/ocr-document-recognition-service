import logging
import os
from typing import List

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator='.core-root', pythonpath=True)

from data.data_classes import Prediction
from pipelines.run import CorePipeline
from utils.metrics import Metrics


def print_metrics(detector_metrics, recognizer_metrics):
    for detector_metric in detector_metrics:
        res = detector_metric.get_metrics()
        logging.info(f'✅ Detection metrics {detector_metric}: {res}')

    print("------------------------------------------------------------------")

    for recognizer_metric in recognizer_metrics:
        res = recognizer_metric.get_metrics()
        logging.info(f'✅ Recognizer metrics {recognizer_metric}: {res}')


@hydra.main(version_base=None, config_path='../config', config_name='evaluate')
def evaluate(cfg: DictConfig):
    print(cfg)
    pipeline: CorePipeline = instantiate(cfg['shift_ocr'])
    logging.info(f'⏰ Evaluating {len(pipeline._loader)} images ⏰')

    detector_metrics: Metrics = instantiate(cfg['detector_metrics'])
    recognizer_metrics: Metrics = instantiate(cfg['recognizer_metrics'])

    print(detector_metrics)
    print(recognizer_metrics)

    count = 0

    for img, gt_detections in tqdm(
        pipeline._loader.load(), total=len(pipeline._loader)
    ):
        img = pipeline._preprocessor.preprocessing(img)

        detections: List[Prediction] = pipeline._detector.detect(img)

        # call detection metrics
        for detector_metric in detector_metrics:
            detector_metric.process(detections, gt_detections)

        detections: List[Prediction] = pipeline._postprocessor.postprocessing(
            detections, img
        )

        recognition: List[Prediction] = pipeline._recognizer.recognize(detections)

        for recognizer_metric in recognizer_metrics:
            recognizer_metric.process(detections, gt_detections)

        if count > 0 and count % cfg.print_every_step == 0:
            print_metrics(detector_metrics, recognizer_metrics)

        count += 1

    print_metrics(detector_metrics, recognizer_metrics)


if __name__ == '__main__':
    evaluate()
