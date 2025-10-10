from typing import Optional

import hydra
import torch
from constansts import DESIRED_T_SIZE
from data.data_classes import Prediction
from doctr.datasets.vocabs import VOCABS
from doctr.models import CRNN, crnn_vgg16_bn, recognition_predictor
from omegaconf import DictConfig, OmegaConf
from pipelines.default.recognizer import Recognizer
from pipelines.recognizer.components.feature_reshape import FeatureWidthInterpolator
from utils.ctc_decoder import MaskedCTCDecoder


class CRNNRecognizer(Recognizer):
    def _init(
        self,
        backbone: DictConfig,
        cuda: bool = True,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        input_shape: Optional[tuple[int, int, int]] = (3, 40, 200),
        vocab: Optional[str] = VOCABS["multilingual"],
        filter_ctc: Optional[bool] = False,
        frames: Optional[int] = 3,
        decoder: str = "lstm",
    ):
        """
        Recognizer CRNN-VGG16

        Args:
            backbone (DictConfig): конфиг для бэкбона, содержит имя и претрейн флаг
            cuda (bool): Использовать GPU
            weights_path (str): Путь до кастомных весов
            pretrained (bool): Предобученная модель
            max_length (int): Максимальная длина последовательности
            vocab (str): Вокабуляр для модели
            filter_ctc (bool): Использовать ли фильтрацию MaskedCTC
            frames (int): Сколько брать выборку для маскирования
        """
        self.cuda = cuda
        self.input_shape = tuple(input_shape)

        if backbone.name == "vgg16":
            crnn_net = crnn_vgg16_bn(
                pretrained=pretrained, vocab=vocab, input_shape=self.input_shape
            )
        else:
            backbone_path = rf"config/core/recognizer/backbones/{backbone.name}.yaml"
            backbone_cfg = OmegaConf.load(backbone_path)

            def build_model(pretrained):
                d_model = hydra.utils.instantiate(backbone_cfg, pretrained=pretrained)
                model = next(iter(d_model.values()))
                backbone = torch.nn.Sequential(
                    *[i[1] for i in model.named_children()],
                    FeatureWidthInterpolator(desired_T=DESIRED_T_SIZE),
                    torch.nn.AdaptiveAvgPool2d((1, None)),
                )
                return backbone

            crnn_net = CRNN(
                feature_extractor=build_model(backbone.pretrained),
                vocab=vocab,
                input_shape=self.input_shape,
            )

            crnn_net.cfg = {
                "mean": [0.694, 0.695, 0.693],
                "std": [0.299, 0.296, 0.301],
                "input_shape": input_shape,
            }

        if filter_ctc:
            crnn_net.postprocessor = MaskedCTCDecoder(detect_frames=5)

        if decoder:
            with torch.no_grad():
                features = crnn_net.feat_extractor(torch.zeros((1, *self.input_shape)))
                out_shape = features.shape
                dec_in = out_shape[1] * out_shape[2]

            decoder_path = rf"config/core/recognizer/decoders/{decoder}.yaml"
            decoder_cfg = OmegaConf.load(decoder_path)
            decoder_cfg.input_size = dec_in
            crnn_net.decoder = hydra.utils.instantiate(decoder_cfg)

            crnn_net.linear = torch.nn.Linear(
                in_features=(1 + int(decoder_cfg.bidirectional))
                * decoder_cfg.hidden_size,
                out_features=len(vocab) + 1,
            )
            for name, param in crnn_net.decoder.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)

        if weights_path:
            weights_state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            crnn_net.load_state_dict(weights_state_dict)

        if cuda and torch.cuda.is_available():
            crnn_net.cuda()
            crnn_net.decoder.cuda()
            crnn_net.feat_extractor.cuda()
            crnn_net.linear.cuda()

        self.net = recognition_predictor(arch=crnn_net, batch_size=1)

    def recognize(self, post_detections) -> list[Prediction]:

        for det in post_detections:
            det.text, det.text_score = self.net([det.crop])[0]

        return post_detections
