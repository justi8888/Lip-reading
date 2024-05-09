# Configuration files
The architecture and behavior of model can be modified using `yaml` configuration files. These files form a structure with one main configuration file and other configuration files specifying details about architecture, dataset, optimizer etc. I provide 8 different main configuration files for different models:
### Simple
- `simple` - simple architecture, character tokenization with diacritics (last linear layer is changed to 57 output classes)
- `simple_nod` - simple architecture, character tokenization without diacritics (41 output classes)
- `simple_added_linear` - simple architecture, character tokenization with diacritics. One more linear layer with 57 classes is added on the top of original linear layer with 41 classes. For this model, changing the output dim of the decoder in the source code is required


Models called simple with architecture adopted from [Visual Speech Recognition for Multiple Languages in the Wild](https://arxiv.org/abs/2202.13084) have:
- `adim` = 256 
- `aheads` = 4 
- `eunits` = 2048
- `ddim` = 256 
- `dheads` = 4
- `dunits` = 2048

### Complex
- `complex` - complex architecture, character tokenization with diacritics (last linear layer is changed to 57 output classes)
- `complex_nod` - complex architecture, character tokenization without diacritics (last linear layer is changed to 41 output classes)
- `complex_unigram` - complex architecture, tokenization using SentecePiece unigram model (last linear layer is changed to 5008 output classes)


Complex models with architecture adopted from [Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels](https://arxiv.org/abs/2303.14307) have:
- `adim` = 768 
- `aheads` = 12 
- `eunits` = 3072 
- `ddim` = 768 
- `dheads` = 12 
- `dunits` = 3072



## Parameters
### Defaults
These default paramaters refer to other configuration files.
- `defaults`:
  - `dataset` - name of the dataset configuration file
  - `model` - name of the configuration file defining the architecture
  - `optimizer` - name of the optimizer configuration file
  - `trainer` - name of the trainer configuration file
  - `decode` - name of the decoding configuration file


### Other parameters
- `max_frames` - maximal number of frames in batch for training
- `max_frames_val` - maximal number of frames in batch for validation
- `output_type` - `char` for character tokenization using diacritics, `char_nod` for character tokenization without diacritics, `token` for SentecePiece unigram tokenization
- `model_type` - either `simple` or `complex`
- `pretrained_model_path` - path to pretrained model
- `reset_last_layer` - resetting the weights of last layer
- `freeze_frontend` - freeze front-end weights during training
- `freeze_encoder` - freeze the whole encoder weights during training
- `gpus` - number of gpus, will be set after the start of the program, according to available gpus on user's machine
- `exp_dir` - parent directory for experiment
- `exp_name` - directory where checkpoints and logs will be saved
- `testing` - set to True for testing
- `ckpt` - path to ckpt for testing 
- `results_csv` - path to the csv file where results of testing will be recorded


All of these paramters can be modified directly in the yaml configuration file or passed as arguments with `train.py` or `eval.py` 





