import random
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from doctr.models import crnn_vgg16_bn, crnn_mobilenet_v3_large, CRNN
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import torch.nn as nn
from torchvision.transforms.v2 import Normalize
from src.models.components.cer_metric import CERMetric
from src.models.components.wer_metric import WERMetric
from .components.ctc_decoder import MaskedCTCDecoder
from .components.feature_reshape import FeatureWidthInterpolator

class CRNNModel(LightningModule):

    def __init__(
        self,
        backbone: str,
        vocab: str,
        timesteps: int,
        pretrained: bool,
        weights_path: str,
        decoder: str,
        dropout: float,
        input_shape: list[int],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cuda: str,
        compile: bool,
        filter_ctc: bool,
        frames: int,
    ):
        """Класс LightningModule для обучения CRNN модели
        
        params:
        :optim_cfg: ка
        """
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.input_shape = tuple(input_shape)
            
        if backbone.name == 'vgg16':
            crnn_net = crnn_vgg16_bn(
                pretrained=pretrained, 
                pretrained_backbone=backbone.pretrained, 
                vocab=vocab, 
                input_shape=self.input_shape
            )
        elif backbone.name == 'mobilenet_v3_large':
            crnn_net = crnn_mobilenet_v3_large(
                pretrained=pretrained, 
                pretrained_backbone=backbone.pretrained, 
                vocab=vocab, 
                input_shape=self.input_shape
            )
        else:
            backbone_path = f'configs/model/backbones/{backbone.name}.yaml'
            backbone_cfg = OmegaConf.load(backbone_path)

            def build_model(pretrained):
                d_model = hydra.utils.instantiate(backbone_cfg, pretrained=pretrained)
                model = next(iter(d_model.values()))
                backbone = nn.Sequential(
                    *[i[1] for i in model.named_children()],
                    FeatureWidthInterpolator(desired_T=40),  
                    nn.AdaptiveAvgPool2d((1, None)),
                )
                return backbone

            crnn_net = CRNN(
                feature_extractor=build_model(backbone.pretrained), 
                vocab=vocab,
                input_shape=self.input_shape
            )

            crnn_net.cfg = {
                'mean': [0.694, 0.695, 0.693],
                'std': [0.299, 0.296, 0.301],
                'input_shape': self.input_shape
            }

        if filter_ctc:
            crnn_net.postprocessor = MaskedCTCDecoder(vocab=vocab, detect_frames=frames)
        
        if decoder:
            with torch.no_grad():
                features = crnn_net.feat_extractor(torch.zeros((1, *self.input_shape)))
                out_shape = features.shape
                print('output shape: ', out_shape)
                dec_in = out_shape[1] * out_shape[2]

            decoder_path = f'configs/model/decoder_type/{self.hparams.decoder}.yaml'
            decoder_cfg = OmegaConf.load(decoder_path)
            decoder_cfg.input_size = dec_in
            decoder_cfg.dropout = self.hparams.dropout
            crnn_net.decoder = hydra.utils.instantiate(decoder_cfg)
            
            crnn_net.linear = nn.Linear(
                in_features= (1 + int(decoder_cfg.bidirectional)) * decoder_cfg.hidden_size, 
                out_features=len(vocab) + 1
            )
            for name, param in crnn_net.decoder.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
            
        if weights_path:
            weights_state_dict = torch.load(
                weights_path, map_location='cpu', weights_only=False
            )
            crnn_net.load_state_dict(weights_state_dict)

        if cuda and torch.cuda.is_available():
            crnn_net.cuda()
            crnn_net.feat_extractor.cuda()
            crnn_net.decoder.cuda()
            hydra.utils.log.info(f'Device is set to CUDA')
            torch.set_float32_matmul_precision('high')

        self.net = crnn_net
        self.net.train()
        
        self.batch_transform = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.cer_val = CERMetric()
        self.wer_val = WERMetric()
        self.best_cer = MinMetric()
        self.best_wer = MinMetric()

        self.cer_test = CERMetric()
        self.wer_test = WERMetric()

    def forward(self, x):
        image, text, _ = x
        images = self.batch_transform(image.to(self.device))
        return self.net(images, text, return_preds=True)

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        self.cer_val.reset()
        self.wer_val.reset()
        self.best_cer.reset()
        self.best_wer.reset()


        self.cer_test.reset()
        self.wer_test.reset()

    def training_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)
        output = self.net(images, labels, return_preds=True)
        loss = output['loss']

        self.train_loss(loss)

        self.log(
            'train/loss', self.train_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        if batch_idx % 5 == 0:
            hydra.utils.log.info(
                f"Обучение. Эпоха: {self.current_epoch}, Лосс: {loss:.4f}, промежуточные результаты:"
            )
            for _ in range(3):
                random_idx = random.randint(0, len(output['preds']) - 1)
                predicted_text = output['preds'][random_idx][0]
                true_text = labels[random_idx]
                hydra.utils.log.info(
                    f"Предсказание='{predicted_text}', истинная метка='{true_text}'"
                )

        return loss

    def on_train_epoch_end(self):
        # Логирование среднего лосса за эпоху
        self.log('train/loss_epoch', self.train_loss.compute(), prog_bar=True)
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)
        output = self.net(images, labels, return_preds=True)

        # Обновление метрик
        loss = output['loss']
        self.val_loss(loss)
        self.cer_val(output['preds'], labels) 
        self.wer_val(output['preds'], labels)

        # Логирование на шаге
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx == 0:
            hydra.utils.log.info(
                f"Валидация. Эпоха: {self.current_epoch}, Лосс: {loss:.4f}, промежуточные результаты:"
            )
            for _ in range(3):
                random_idx = random.randint(0, len(output['preds']) - 1)
                predicted_text = output['preds'][random_idx][0]
                true_text = labels[random_idx]
                hydra.utils.log.info(
                    f"Предсказание='{predicted_text}', истинная метка='{true_text}'"
                )

    def on_validation_epoch_end(self) -> None:
        cer_val = self.cer_val.compute()
        wer_val = self.wer_val.compute()
        loss_val = self.val_loss.compute()

        val_h_mean = 2 * cer_val * wer_val / (cer_val + wer_val + 1e-8)

        self.best_cer.update(cer_val)
        self.best_wer.update(wer_val)

        self.log_dict({
            'val/loss': loss_val,
            'val/cer': cer_val,
            'val/wer': wer_val,
            'val/best_cer': self.best_cer.compute(),
            'val/best_wer' : self.best_wer.compute(),
            'val/harmonic_mean' : val_h_mean,
        }, prog_bar=True)

        self.val_loss.reset()
        self.cer_val.reset()
        self.wer_val.reset()

    def test_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)
        output = self.net(images, labels, return_preds=True)

        # Обновление метрик
        loss = output['loss']
        self.test_loss(loss)
        self.cer_test(output['preds'], labels)
        self.wer_test(output['preds'], labels)
        
        # Логирование
        self.log('test/loss', self.test_loss, on_step=True, on_epoch=False, prog_bar=True)

        # Вывод всех неправильных предсказаний
        incorrect_predictions = []
        for _, (pred, true) in enumerate(zip(output['preds'], labels)):
            predicted_text = pred[0]  # Предполагаем, что pred имеет формат [['text1'], ['text2'], ...]
            if predicted_text != true:
                incorrect_predictions.append((predicted_text, true))

        # Логируем только если есть ошибки (чтобы не засорять вывод)
        if incorrect_predictions:
            for pred_text, true_text in incorrect_predictions:
                hydra.utils.log.info(
                    f"Тест: предсказание='{pred_text}', истинная метка='{true_text}'"
                )
            
            # Дополнительно выводим статистику по батчу
            total = len(output['preds'])
            incorrect = len(incorrect_predictions)
            hydra.utils.log.info(
                f"Ошибки в батче: {incorrect}/{total} ({incorrect/total:.1%})"
            )

    def on_test_epoch_end(self):
        test_loss = self.test_loss.compute()
        test_cer = self.cer_test.compute()
        test_wer = self.wer_test.compute()

        h_mean = 2 * test_cer * test_wer / (test_cer + test_wer + 1e-8)

        self.log_dict({
            'test/loss': test_loss,
            'test/cer': test_cer,
            'test/wer': test_wer,
            'test/harmonic_mean' : h_mean,
        }, prog_bar=True)

        self.test_loss.reset()
        self.cer_test.reset()
        self.wer_test.reset()

        
    def on_train_end(self) -> None:
        """Хук, который вызывается после завершения всего процесса обучения."""
        save_dir = '../weights/recognizer/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{self.hparams.backbone.name}_{self.hparams.decoder}_best.pth')
        torch.save(self.net.state_dict(), save_path)
        hydra.utils.log.info(f"Модель успешно сохранена: {save_path}")


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == 'fit':
            self.net = torch.compile(self.net)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        params = [p for p in self.net.parameters() if p.requires_grad]

        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'step',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
