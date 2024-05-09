import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer
from datamodule.data_module import DataModule
from lightning import ModelModule
from ckpt_to_pth import ckpt_to_pth


@hydra.main(version_base="1.3", config_path="configs", config_name="simple")
def main(cfg):
    #cfg.gpus = 1
    cfg.gpus = torch.cuda.device_count()
    
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    #trainer = Trainer(num_nodes=1, gpus=1)
    trainer = Trainer(num_nodes=1, gpus=2, strategy='ddp')
    
    # load checkpoint from config file and transform it to model pth fie
    ckpt = cfg.ckpt
    pth = ckpt_to_pth(ckpt=ckpt)
    modelmodule.model.load_state_dict(pth, strict=True)
    
    #when loading pth model 
    #modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
    trainer.test(model=modelmodule, datamodule=datamodule)
    


if __name__ == "__main__":
    main()
