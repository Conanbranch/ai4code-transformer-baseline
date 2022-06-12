# ai4code-transformer-baseline

Solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition

## Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and Khoi Nguyen's [baseline](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells). 

Instead of predicting the rank of the markdown cells, code cells are sampled from the notebook to provide context for the markdown. The code cells are sampled uniformaly, perserving the order. Input will look like this: 

```<CLS> markdown <SEP> code <SEP> code <SEP> ... <SEP> code <SEP>```

Ranking of code cells is preserved between training and validation sets:

- Add

## Preprocess
To prepare the markdown and code cells for training run:

```$ python preprocess.py```

To reduce the size of the data set used for training and validation:

```$ python preprocess.py --sample_data 0.2```

This will create training and validation set that is 20% of the data with a 90/10 split (i.e the training set will be 18% and the validation set is 2%)

Output will be in the ```./data``` folder:
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

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 2```

To continue from previous checkpoint:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 2 --resume_train True```

To continue from previous checkpoint (with specific filename name and path):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 2 --resume_train True --model_ckp_path "/checkpoint_path" --model_ckp "checkpoint.pt"```

To save model (with specific file name):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 2 --model "model.bin"```

### Training Time

~ 20 Hours - Tesla A100

~ 33 Hours - Tesla V100-SXM2-16GB

## Inference
- Add

## To Do

- Clean up ranking
- Clean up default command line arguments
- Add Stochastic Weight Averaging (SWA)
- Maybe add frequent evaluation
- double check on named paramaters in weight decay
- confirm no that the notebooks with the same parent ID have the same ancestor ID
- May want to sweep batch size depending on GPU availailibity 8, 16 (with V100 or lower), 32, 64 (with A100)
- May want to sweep learning rates
- Clean up input (remove comments from code, remove markup and other stuff from comments)
- Sweep # of epochs try 3 - 10
- Could also try using an activation function and dropout
- Could also try to use MSE instead of BCE
- Look into end of sentence token [EOS]
- May want to do a paramater sweep with a smaller md_max_len such as --md_max_len 32 for different code sample sizes, larger sample sizes may simply be impacted by the size of the each code sample as it will get smaller with more samples. 
- Might want to check in reseting the pooler
