# Neural Machine Translation

## Overview
### Dataset
I used the dataset [PhoMT](https://github.com/VinAIResearch/PhoMT) from VinAIResearch. PhoMT is a high-quality and large-scale Vietnamese-English parallel dataset of 3.02M sentence pairs. For my project, I used training consist of 400K of 2.9M sentence pairs from `train.en` and `train.vi`. The validation set is taken from `dev.en` and `dev.vi` (15K pairs), test set is from `test.en` and `test.vi` (almost 20K pairs).

### Seq2seq model
It is a Encoder-Decoder architecture, in which the Encoder takes the input features and passes its output to the first RNN unit of the Decoder. The Decoder will generate sentences word by word. Here is a simple illustration of the Seq2seq model:


## Run project
1. Install required packages: `$ ./run_build.sh` or `$ bash run_build.sh`
2. Edit project's configuration in [config.py](config.py)
3. Normalized dataset: `$ python normalize_data.py`
4. For training: `$ python train.py`
5. For testing (evaluate metrics): `$ python test.py`
6. Using trained model to translate: `translate(texts)` in file [translate.py](translate.py)