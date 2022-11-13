# ai4code-transformer-baseline

The following repository is my solution for the [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition. 

I won silver and placed 37/1135 in the competition (top 4%). 

## Overview
This solution is based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and Khoi Nguyen's [baseline](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells) with numerous modifications to the model and data preprocessing. It uses a pretrained [codeBERT](https://github.com/microsoft/CodeBERT) transformer, fine tuned to predict the ranks of markdown cells in each notebook, utilizing code cells sampled from the notebook of the associated markdown cell as additional context. 

The code cells are sampled so they are evenly spaced (aside from the second to last and last cell) and the order of the code cells are preserved, while the first and last code cell are always included.

The input takes the following form: 

```<CLS> markdown <SEP> code_1 <SEP> code_2 <SEP> ... code_n <SEP>```

### Key Additions and Modifications

-	Variable length code for the code samples (i.e. if there were only 7 code cells to sample then more code from those cells are sampled)
-	Ranking is preserved between training, validation and test sets for code cells
-	Sigmoid top for the model
-	Layer re-initialization for top layer of the transformer
-	No limits on the size of the code cell being sampled (aside from the max length available for each code cell)
-	Preprocessing to remove newlines
-	Some options to play around with code padding and separators
-	Multi-GPU support

This is not an exhaustive list of all modifications and additions.

## Preprocess
To prepare the markdown and code cells for training run:

```$ python preprocess.py```

This will generate a 90/10 split for training/dev

To reduce the size of the data set used for training and validation:

```$ python preprocess.py --sample_data 0.2```

This will create training and validation set that is 20% of the data with a 90/10 split (i.e the training set will be 18% and the validation set is 2%)

To change the number of feature samples:

```$ python preprocess.py --num_samples 40```

The above command will produce the following output:

```
ai4code-transformer-baseline/data
│   train_mark.csv
│   train_fts.json   
|   train.csv
│   val_mark.csv
│   val_fts.json
│   val.csv
```

## Train 

To train the model: 

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2```

To continue from previous checkpoint:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --resume_train```

To continue from previous checkpoint (with specific filename name and path):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --resume_train --model_ckp_path "/checkpoint_path" --model_ckp "checkpoint.pt"```

To save model (with specific file name):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --model "model.bin"```

The above command will produce the following output:

```
ai4code-transformer-baseline/output
|   epoch_1_model.bin
│   epoch_2_model.bin
│   epoch_3_model.bin
│   ...
|   epoch_n_model.bin
|   model.bin
|   model.pt
```

### Variable Length Code

To use variable length code:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 2 --vbl_code```

The default is ```code_max_length = int((self.total_max_len - self.md_max_len)/num_samples) + 1 ```, where the code size (number of tokens per code cell) is fixed regardless of the number of code cells in the notebook. With variable length code, ```code_max_length = int((self.total_max_len - self.md_max_len)/num_sampled) + 1``` and is a function of the actual number of code cells sampled (i.e. there were only 7 code cells ```num_sampled = 7``` but ```num_samples = 20```, this gives you more tokens per code cells when ```num_sampled < num_samples```).

### Training Time (Per Model)

~ 40 Hours - A100-SXM4-40GB - 10 Epochs - 54.8 Gb RAM

~ 66 Hours - V100-SXM2-16GB - 10 Epochs - 54.8 Gb RAM

## Reproducing Submitted Models

For my final solution I combined 3 different trained models via a weighted ensemble:

Model 1:

```$ python preprocess.py --num_samples 40```

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 3e-5 --wd 0.1 --vbl_code --pad_between_code --code_sep_token```

Model 2:

```$ python preprocess.py --num_samples 40```

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 3e-5 --wd 0.01 --pad_between_code --code_sep_token```

Model 3:

```$ python preprocess.py --num_samples 40```

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 12 --re_init --reinit_n_layers 1 --lr 4e-5 --wd 0.01 --pad_between_code --code_sep_token```

Weighted Ensemble:

- To Do

Example Notebooks: 

- To Do

## Results - Trained on 90% of Dataset

| Model | Val | Public | Private |
| --- | --- | --- | --- |
| Model 1 | .8676 | .8610  | NA |
| Model 2 | .8650 | .8575 | NA |  
| Model 3 | .8641 | .8567 | NA |
| Weighted Ensemble | .8715 | .8643 | .8638

Weights for the weighted ensemble were 1.0, 0.6, and 0.6 corresponding to model 1, 2, and 3.
