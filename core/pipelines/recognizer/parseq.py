import os
from typing import Optional

import hydra
import rootutils
import torch
from data.data_classes import Prediction

from doctr.datasets.vocabs import VOCABS
from doctr.models import PARSeq, parseq, recognition_predictor
from omegaconf import DictConfig, OmegaConf
from pipelines.default.recognizer import Recognizer
from pipelines.recognizer.components.feature_reshape import FeatureWidthInterpolator
from utils.ctc_decoder import MaskedCTCDecoder



class PARSeqRecognizer(Recognizer):
    def _init(
        self,
        cuda: bool = True,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        input_shape: Optional[tuple[int, int, int]] = (3, 32, 128),
        vocab: Optional[str] = VOCABS["multilingual"],
    ):
        """
        Recognizer PARSeq из библиотеки docTR

        Args:
            cuda (bool): Использовать GPU
            weights_path (str): Путь до кастомных весов
            pretrained (bool): Предобученная модель
            vocab (str): Вокабуляр для модели
        """
        self.cuda = cuda
        self.input_shape = tuple(input_shape)
        
        parseq_net = parseq(
            pretrained=pretrained,
            input_shape=input_shape,
            vocab=vocab,
        )
        
        if weights_path:
            weights_state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            parseq_net.load_state_dict(weights_state_dict)

        if cuda and torch.cuda.is_available():
            parseq_net.cuda()

        self.net = recognition_predictor(arch=parseq_net, batch_size=1)

    def recognize(self, post_detections) -> list[Prediction]:

        for det in post_detections:
            det.text, det.text_score = self.net([det.crop])[0]

        return post_detections
