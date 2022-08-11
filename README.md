# ai4code-transformer-baseline

My solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition.

## Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and Khoi Nguyen's [baseline](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells) with several modifications to the model and preprocessing. 

Code cells are sampled from the notebook to provide context for the markdown. The code cells are sampled uniformaly, perserving the order. Input will look like this: 

```<CLS> markdown <SEP> code_1 <SEP> code_2 <SEP> ... code_n <SEP>```

### Important Additions

- multi-gpu support
- variable length code for the code samples (i.e. if there was only 7 code cells to sample then more code from those cells is sampled)
- ranking perserved between training, validation and test sets for code cells and forces values to between 0 and 1
- sigmoid top for the model
- layer re-itialization for top layer of transfomer
- no limit on the size of the code cell being sampled
- preprocessing that removes newlines
- removed "fts" from MarkdownModel
- some options to play around with code padding and seperators

This is not an extensive list of all modifications

## Preprocess
To prepare the markdown and code cells for training run:

```$ python preprocess.py```

This will generate a 90/10 split for training/dev

To reduce the size of the data set used for training and validation:

```$ python preprocess.py --sample_data 0.2```

This will create training and validation set that is 20% of the data with a 90/10 split (i.e the training set will be 18% and the validation set is 2%)

To change the number of feature samples:

```$ python preprocess.py --num_samples 40```

```
ai4code-transformer-baseline
│   train_mark.csv
│   train_fts.json   
|   train.csv
│   val_mark.csv
│   val_fts.json
│   val.csv
```

## Train

This solution fine tunes the code-bert pre-trained transformer. To fine tune the transfomer: 

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2```

To continue from previous checkpoint:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --resume_train```

To continue from previous checkpoint (with specific filename name and path):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --resume_train --model_ckp_path "/checkpoint_path" --model_ckp "checkpoint.pt"```

To save model (with specific file name):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --model "model.bin"```

Output will be in the ```./data``` folder:

### Variable Length Code

To use variable length code:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --vbl_code```

The default is ```code_max_length = int((self.total_max_len - self.md_max_len)/num_samples)```, where the code size (number of tokens per code cell) is fixed regardless of the number of code cells in the notebook. With variable length code, ```code_max_length = int((self.total_max_len - self.md_max_len)/num_sampled)``` and is a function of the actual number of code cells sampled (i.e. there were only 7 code cells ```num_sampled = 7``` but ```num_samples = 20```, this gives you more tokens per code cells when ```num_sampled < num_samples```).

### Training Time

~ 40 Hours - Tesla A100-SXM4-40GB - 10 Epochs - 54.8 Gb RAM

~ 66 Hours - Tesla V100-SXM2-16GB - 10 Epochs - 54.8 Gb RAM

## Reproducing Final Models

For my final solution I combined 3 different trained models via an average ensemble:

Model 1:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 3e-5 --wd 0.1 --vbl_code --pad_between_code --code_sep_token```

Model 2:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 3e-5 --wd 0.01 --pad_between_code --code_sep_token```

Model 3:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 4e-5 --wd 0.01 --pad_between_code --code_sep_token```

## Results - 90% Dataset Eval

Add

## To Do

- Add models and data
