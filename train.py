import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from datamodule.data_module import DataModule
from pytorch_lightning.loggers import WandbLogger
from lightning import ModelModule


@hydra.main(version_base="1.3", config_path="configs", config_name="simple")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()
    #cfg.gpus = 0


    checkpoint = ModelCheckpoint(
        monitor="wer",
        mode="min",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=2,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    logger=WandbLogger(name=cfg.exp_name, project="Experiments")
    
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=1
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
