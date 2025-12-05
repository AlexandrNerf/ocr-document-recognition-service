import os
from typing import Optional

import hydra
import rootutils
import torch
from data.data_classes import Prediction

# Константа для интерполяции ширины фичей
DESIRED_T_SIZE = 200
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
        
        # Определяем корень core директории
        # Используем путь относительно текущего файла (надежнее)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Поднимаемся из pipelines/recognizer/ в core/
        # __file__ = core/pipelines/recognizer/crnn.py
        # dirname 3 раза: core/pipelines/recognizer -> core/pipelines -> core -> (корень проекта)
        # Но нам нужен core, поэтому dirname 2 раза
        core_root = os.path.dirname(os.path.dirname(current_file_dir))
        
        # Проверяем, что путь правильный
        config_path = os.path.join(core_root, "config")
        if not os.path.exists(config_path):
            # Пробуем альтернативный путь через rootutils
            try:
                project_root = rootutils.find_root(indicator=".core-root")
                core_config_path = os.path.join(project_root, "core", "config")
                if os.path.exists(core_config_path):
                    core_root = os.path.join(project_root, "core")
                    config_path = core_config_path
                else:
                    raise FileNotFoundError(
                        f"Не найдена директория config. "
                        f"Проверенные пути: {os.path.join(core_root, 'config')}, {core_config_path}"
                    )
            except Exception as e:
                raise FileNotFoundError(
                    f"Не найдена директория config в {core_root}. "
                    f"Текущий файл: {__file__}, проверяемый путь: {config_path}. "
                    f"Ошибка rootutils: {e}"
                )

        if backbone.name == "vgg16":
            crnn_net = crnn_vgg16_bn(
                pretrained=pretrained, vocab=vocab, input_shape=self.input_shape
            )
        else:
            backbone_path = os.path.join(
                core_root, "config", "core", "recognizer", "backbones", f"{backbone.name}.yaml"
            )
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

            decoder_path = os.path.join(
                core_root, "config", "core", "recognizer", "decoders", f"{decoder}.yaml"
            )
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
            # Делаем путь абсолютным, если он относительный
            if not os.path.isabs(weights_path):
                # Если путь начинается с ../, значит он относительно корня проекта
                if weights_path.startswith("../"):
                    # Поднимаемся из core/ в корень проекта
                    project_root = os.path.dirname(core_root)
                    weights_path = os.path.join(project_root, weights_path[3:])  # Убираем ../
                else:
                    # Относительный путь от core_root
                    weights_path = os.path.join(core_root, weights_path)
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Файл весов не найден: {weights_path}. "
                    f"Проверьте путь в конфиге."
                )
            
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
        if not hasattr(self, 'net') or self.net is None:
            raise RuntimeError(
                "CRNNRecognizer не был правильно инициализирован. "
                "Проверьте логи на наличие ошибок при инициализации."
            )

        for det in post_detections:
            det.text, det.text_score = self.net([det.crop])[0]

        return post_detections
