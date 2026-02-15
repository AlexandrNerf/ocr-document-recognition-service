# import random
# import os
# import hydra
# import torch
# from omegaconf import DictConfig, OmegaConf
# from doctr.models import PARSeq, parseq
# from lightning import LightningModule
# from torchmetrics import MaxMetric, MeanMetric, MinMetric
# import torch.nn as nn
# from torchvision.transforms.v2 import Normalize
# from src.models.components.cer_metric import CERMetric
# from src.models.components.wer_metric import WERMetric
# from .components.ctc_decoder import MaskedCTCDecoder
# from .components.feature_reshape import FeatureWidthInterpolator

# class PARSeqModel(LightningModule):

#     def __init__(
#         self,
#         vocab: str,
#         pretrained: bool,
#         weights_path: str,
#         input_shape: list[int],
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler,
#         cuda: str,
#         compile: bool,
#     ):
#         """Класс LightningModule для обучения CRNN модели
        
#         params:
#         :optim_cfg: ка
#         """
#         super().__init__()
#         self.save_hyperparameters(logger=True)

#         self.input_shape = tuple(input_shape)

#         parseq_net = parseq(
#             pretrained=pretrained,
#             input_shape=self.input_shape,
#             vocab=vocab,
#         ) 

#         if weights_path:
#             weights_state_dict = torch.load(
#                 weights_path, map_location='cpu', weights_only=False
#             )
#             parseq_net.load_state_dict(weights_state_dict)

#         if cuda and torch.cuda.is_available():
#             parseq_net.cuda()
#             parseq_net.feat_extractor.cuda()
#             parseq_net.decoder.cuda()
#             hydra.utils.log.info(f'Device is set to CUDA')
#             torch.set_float32_matmul_precision('high')

#         self.net = parseq_net
#         self.net.train()
        
#         self.batch_transform = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

#         self.train_loss = MeanMetric()
#         self.val_loss = MeanMetric()
#         self.test_loss = MeanMetric()

#         self.cer_val = CERMetric()
#         self.wer_val = WERMetric()
#         self.best_cer = MinMetric()
#         self.best_wer = MinMetric()

#         self.cer_test = CERMetric()
#         self.wer_test = WERMetric()
#         self.cer_train = CERMetric()
#         self.wer_train = WERMetric()

#     def forward(self, x):
#         image, text, _ = x
#         images = self.batch_transform(image.to(self.device))
#         return self.net(images, text, return_preds=True)

#     def on_train_start(self) -> None:
#         self.train_loss.reset()
#         self.val_loss.reset()
#         self.test_loss.reset()

#         self.cer_val.reset()
#         self.wer_val.reset()
#         self.best_cer.reset()
#         self.best_wer.reset()

#         self.cer_train.reset()
#         self.wer_train.reset()
#         self.cer_test.reset()
#         self.wer_test.reset()

#     def training_step(self, batch, batch_idx):
#         imgs, labels, _ = batch
#         images = self.batch_transform(imgs)
#         output = self.net(images, labels, return_preds=False)
#         loss = output['loss']

#         self.log(
#             "train/loss",
#             loss,
#             on_step=True,
#             on_epoch=False,
#             prog_bar=True,
#             batch_size=images.size(0),
#         )

#         return loss

#     def on_train_epoch_end(self):
#         # Логирование среднего лосса за эпоху
#         self.log('train/loss_epoch', self.train_loss.compute(), prog_bar=True)
#         self.train_loss.reset()

#     def on_train_batch_end(self, outputs, batch, batch_idx):
#         opt = self.optimizers()
#         lr = opt.param_groups[0]["lr"]
#         self.log("lr", lr, on_step=True, prog_bar=False)

#     def validation_step(self, batch, batch_idx):
#         imgs, labels, _ = batch
#         images = self.batch_transform(imgs)

#         out = self.net(images, labels, return_preds=True)

#         loss, preds = out["loss"], out["preds"]

#         self.val_loss.update(loss)
#         self.cer_val(preds, labels)
#         self.wer_val(preds, labels)

#         self.log("val/loss", loss, prog_bar=True)
#         self.log("val/cer", self.cer_val, prog_bar=True)
#         self.log("val/wer", self.wer_val, prog_bar=True)

#         if batch_idx == 0:
#             self._log_predictions(images, labels, preds, stage="val")

#     def on_validation_epoch_end(self) -> None:
#         cer_val = self.cer_val.compute()
#         wer_val = self.wer_val.compute()
#         loss_val = self.val_loss.compute()

#         val_h_mean = 2 * cer_val * wer_val / (cer_val + wer_val + 1e-8)

#         self.best_cer.update(cer_val)
#         self.best_wer.update(wer_val)

#         self.log_dict({
#             'val/loss': loss_val,
#             'val/cer': cer_val,
#             'val/wer': wer_val,
#             'val/best_cer': self.best_cer.compute(),
#             'val/best_wer' : self.best_wer.compute(),
#             'val/harmonic_mean' : val_h_mean,
#         }, prog_bar=True)

#         self.val_loss.reset()
#         self.cer_val.reset()
#         self.wer_val.reset()

#     def test_step(self, batch, batch_idx):
#         imgs, labels, _ = batch
#         images = self.batch_transform(imgs)
#         output = self.net(images, labels, return_preds=True)

#         # Обновление метрик
#         loss = output['loss']
#         self.test_loss(loss)
#         self.cer_test(output['preds'], labels)
#         self.wer_test(output['preds'], labels)
        
#         self.log('test/loss', self.test_loss, on_step=True, on_epoch=False, prog_bar=True)
#         self.log('test/cer', self.cer_test, on_step=True, on_epoch=False, prog_bar=True)
#         self.log('test/wer', self.wer_test, on_step=True, on_epoch=False, prog_bar=True)

#     def on_test_epoch_end(self):
#         test_loss = self.test_loss.compute()
#         test_cer = self.cer_test.compute()
#         test_wer = self.wer_test.compute()

#         h_mean = 2 * test_cer * test_wer / (test_cer + test_wer + 1e-8)

#         self.log_dict({
#             'test/loss': test_loss,
#             'test/cer': test_cer,
#             'test/wer': test_wer,
#             'test/harmonic_mean': h_mean,
#         }, prog_bar=True)

#     def _log_predictions(
#         self,
#         images,
#         labels,
#         preds,
#         stage="val",
#         max_items=4,
#     ):
#         if not isinstance(self.logger.experiment, object):
#             return

#         tb = self.logger.experiment

#         images = images[:max_items].detach().cpu()
#         labels = labels[:max_items]
#         preds = preds[:max_items]

#         for i in range(len(images)):
#             img = images[i]
#             pred = preds[i][0]
#             gt = labels[i]

#             tb.add_image(
#                 f"{stage}/image_{i}",
#                 img,
#                 global_step=self.global_step,
#             )
#             tb.add_text(
#                 f"{stage}/text_{i}",
#                 f"GT: {gt} | PRED: {pred}",
#                 global_step=self.global_step,
#             )

        
#     def on_train_end(self) -> None:
#         """Хук, который вызывается после завершения всего процесса обучения."""
#         save_dir = '../weights/recognizer/'
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f'model_{self.hparams.backbone.name}_{self.hparams.decoder}_best.pth')
#         torch.save(self.net.state_dict(), save_path)
#         hydra.utils.log.info(f"Модель успешно сохранена: {save_path}")


#     def setup(self, stage: str) -> None:
#         """Lightning hook that is called at the beginning of fit (train + validate), validate,
#         test, or predict.

#         This is a good hook when you need to build models dynamically or adjust something about
#         them. This hook is called on every process when using DDP.

#         :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
#         """
#         if self.hparams.compile and stage == 'fit':
#             self.net = torch.compile(self.net)

#     def on_after_backward(self):
#         torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5)

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.

#         Examples:
#             https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

#         :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
#         """
#         params = [p for p in self.net.parameters() if p.requires_grad]

#         optimizer = self.hparams.optimizer(params=params)
#         if self.hparams.scheduler is not None:
#             scheduler = self.hparams.scheduler(optimizer=optimizer)
#             return {
#                 'optimizer': optimizer,
#                 'lr_scheduler': {
#                     'scheduler': scheduler,
#                     'monitor': 'val/loss',
#                     'interval': 'step',
#                     'frequency': 1,
#                 },
#             }
#         return {'optimizer': optimizer}



import os
import numpy as np
from PIL import Image, ImageDraw
import hydra
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchvision.transforms.v2 import Normalize

from doctr.models import parseq
from src.models.components.cer_metric import CERMetric
from src.models.components.wer_metric import WERMetric

torch.set_float32_matmul_precision('high')

class PARSeqModel(LightningModule):
    def __init__(
        self,
        vocab: str,
        pretrained: bool,
        weights_path: str | None,
        input_shape: list[int],
        optimizer,
        scheduler,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.input_shape = tuple(input_shape)

        self.net = parseq(
            pretrained=pretrained,
            input_shape=self.input_shape,
            vocab=vocab,
        )

        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.net.load_state_dict(state_dict, strict=False)

        self.batch_transform = Normalize(
            mean=(0.694, 0.695, 0.693),
            std=(0.299, 0.296, 0.301),
        )

        # --- losses ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # --- metrics ---
        self.cer_train = CERMetric()
        self.wer_train = WERMetric()
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

    def training_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)

        out = self.net(images, labels, return_preds=False)
        loss = out["loss"]

        self.train_loss.update(loss)

        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        if batch_idx % 10 == 0:
            opt = self.optimizers()
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=False)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                self.net.eval()
                out_val = self.net(images, labels, return_preds=True)
                preds = out_val["preds"]
                self.net.train()
                # логирование предсказаний
                self._log_predictions(images, labels, preds, stage="train", max_items=6)

                # можно дополнительно посчитать CER/WER на этом батче
                cer = self.cer_train(preds, labels)
                wer = self.wer_train(preds, labels)
                self.log_dict({
                    "train/cer_sample": cer,
                    "train/wer_sample": wer,
                }, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.log(
            "train/loss",
            self.train_loss.compute(),
            prog_bar=True,
        )
        self.train_loss.reset()


    def validation_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)

        out = self.net(images, labels, return_preds=True)
        loss, preds = out["loss"], out["preds"]

        self.val_loss.update(loss)
        self.cer_val(preds, labels)
        self.wer_val(preds, labels)

        if batch_idx == 0:
            self._log_predictions(images, labels, preds, stage="val")

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        cer = self.cer_val.compute()
        wer = self.wer_val.compute()

        h_mean = 2 * cer * wer / (cer + wer + 1e-8)

        self.best_cer.update(cer)
        self.best_wer.update(wer)

        self.log_dict(
            {
                "val/loss": loss,
                "val/cer": cer,
                "val/wer": wer,
                "val/best_cer": self.best_cer.compute(),
                "val/best_wer": self.best_wer.compute(),
                "val/harmonic_mean": h_mean,
            },
            prog_bar=True,
        )

        self.val_loss.reset()
        self.cer_val.reset()
        self.wer_val.reset()

    def test_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        images = self.batch_transform(imgs)

        out = self.net(images, labels, return_preds=True)

        self.test_loss.update(out["loss"])
        self.cer_test(out["preds"], labels)
        self.wer_test(out["preds"], labels)

    def on_test_epoch_end(self):
        loss = self.test_loss.compute()
        cer = self.cer_test.compute()
        wer = self.wer_test.compute()

        h_mean = 2 * cer * wer / (cer + wer + 1e-8)

        self.log_dict(
            {
                "test/loss": loss,
                "test/cer": cer,
                "test/wer": wer,
                "test/harmonic_mean": h_mean,
            },
            prog_bar=True,
        )

        self.test_loss.reset()
        self.cer_test.reset()
        self.wer_test.reset()

    def _log_predictions(self, images, labels, preds, stage="val", max_items=4):
        if not hasattr(self.logger, "experiment"):
            return

        def draw_text_on_image(img_tensor, gt, pred):
            # img_tensor: C,H,W, float [0,1]
            img = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            # font = ImageFont.truetype("arial.ttf", 15)  # можно выбрать шрифт
            draw.text((5, 5), f"GT: {gt} | PRED: {pred}", fill=(255, 0, 0))
            return torch.tensor(np.array(img_pil)).permute(2,0,1) / 255.0  # вернуть в тензор

        tb = self.logger.experiment
        images = images[:max_items].detach().cpu()
        labels = labels[:max_items]
        preds = preds[:max_items]
        for i, (img, gt, pred) in enumerate(zip(images, labels, preds)):
            img_n = self._denormalize(img)
            tb.add_image(
                f"{stage}/image_{i}",
                img_n,
                global_step=self.global_step,
            )
            tb.add_text(
                f"{stage}/text_{i}_{self.global_step}", 
                f"GT: {gt} | PRED: {pred[0]}", 
                global_step=self.global_step
            )

    def _denormalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.694, 0.695, 0.693], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.299, 0.296, 0.301], device=img.device).view(3, 1, 1)
        img = img * std + mean
        return img.clamp(0, 1)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.net.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                },
            }

        return optimizer

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
