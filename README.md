# Neural Machine Translation

## Overview
### Dataset
I used the dataset [PhoMT](https://github.com/VinAIResearch/PhoMT) from VinAIResearch. PhoMT is a high-quality and large-scale Vietnamese-English parallel dataset of 3.02M sentence pairs. For my project, I used training consist of 1M of 2.9M sentence pairs from `train.en` and `train.vi`. The validation set is taken from `dev.en` and `dev.vi` (15K pairs), test set is from `test.en` and `test.vi` (almost 20K pairs).

### Seq2seq model
- Using RNN 
- Orinial: Reference: [Google Tensorflow Example](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
### Modify:
#### Encoder
![encoder.png](pictures/encoder.png)
#### Attention
![attention.png](pictures/attention.png)
#### Decoder
coming soon

## Training
- Optimizer: Adam optimizer
- Leaning rate: initialize 1e-3, and decreased by 0.1 times every epoch
- ```
  history = model.fit(train_ds.repeat(), 
                      epochs=50, 
                      steps_per_epoch = 2500, 
                      validation_data=val_ds, 
                      validation_steps=100), 
                      callbacks=[early_stopping, checkpoint])```
- In training phase, 

## Inference

## Result
### BLEU Score
- I use Google Colaboratory to train and test model, so i will update BLEU metrics into this project

Model | BLEU
:---: | :---:
RNN | 19.425
- **Note:** I still update model until my model get best score

### Examples


## Run project
1. Install required packages: `$ ./run_build.sh` or `$ bash run_build.sh`
2. Edit project's configuration in [config.py](config.py)
3. Normalized dataset: `$ python normalize_data.py`
4. For training: `$ python train.py`
5. For testing (evaluate metrics): `$ python test.py`
6. Using trained model to translate: `translate(texts)` in file [translate.py](translate.py)
