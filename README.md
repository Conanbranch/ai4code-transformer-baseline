# ai4code-transformer-baseline

Solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition

### Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and https://github.com/suicao/ai4code-baseline. 

Instead of predicting the cell position with only the markdown itself, we randomly sample code cells to act as the context. The code is sampled uniformaly, perserving the order. Input will look like this:

```<CLS> markdown <SEP> code <SEP> code <SEP> ... <SEP> code <SEP>```

Ranking of code cells is preserved between training and validation sets:

- Add

Modiified original input from (https://github.com/suicao/ai4code-baseline):

```<CLS> markdown <CLS> code <CLS> code <CLS> ... <CLS> code <CLS> ```

To:

```<CLS> markdown <SEP> code <SEP> code <SEP> ... <SEP> code <SEP>```

### Preprocessing
To prepare the data and extract features for training, including the markdown-only dataframes and sampling the code cells needed for each note book, simply run:

```$ python preprocess.py```

To reduce the size of the data set used for training and validation:

```$ python preprocess.py --sample_data 0.2```

This will create training and validation set that is 20% of the data with a 90/10 split (i.e the training set will be 18% and the validation set is 2%)

Output will be in the ```./data``` folder:
```
project
│   train_mark.csv
│   train_fts.json   
|   train.csv
│   val_mark.csv
│   val_fts.json
│   val.csv
```

###  Training
I found ```codebert-base``` to be the best of all the transformers:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 8```

To continue from previous checkpoint:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 8 --resume_train True```

To continue from previous checkpoint (with specific filename name and path):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 8 --resume_train True --model_ckp_path "/checkpoint_path" --model_ckp "checkpoint.pt"```

To save model (with specific file name):

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 8 --model "model.bin"```

### Inference
- Add

### To Do

- Clean up ranking
- Clean up default command line arguments
- Add Stochastic Weight Averaging (SWA)
- Maybe add frequent evaluation
- may want to try correct_bias=True in adamW as per https://arxiv.org/pdf/2006.05987.pdf
- double check on named paramaters in weight decay
- confirm no that the notebooks with the same parent ID have the same ancestor ID
