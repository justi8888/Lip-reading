## Training and evaluation
For training and evaluation you can use configuration files explained in [configs](./configs). All of these parameters can be specified either in configuration files or passed as parameters during training or evaluation.
In the trainer default parameter in main config file specify either: 
- `train` - training on available GPUs
- `train_cpu` - training on CPU, suitable for debugging 


For training use:
```
python train.py
```
You can specify trainer parameters like: 
- `epochs` - number of epochs
- `resume_from_checkpoint` - path to checkpoint, from which training will continue


For evaluation:
```
python eval.py
```

- `testing` - True - indicates the testing process for correct loading of the models
- `ckpt` - path to checkpoint that will be converted to pth model and further used


