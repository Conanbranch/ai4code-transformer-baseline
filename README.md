# ai4code-transformer-baseline

Solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition

### Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and https://github.com/suicao/ai4code-baseline. 

Instead of predicting the cell position with only the markdown itself, we randomly sample code cells to act as the global context. So your input will look something like this:

```<cls> Markdown content <sep> Code content 1 <sep> Code content 2 <sep> ... <sep> Code content 20 <sep> ```

Ranking of code cells is preserved between training and validation sets:

- Add

Modiified original input from:

```<cls> markdown <cls> code <cls> code <cls> ... <cls> code <cls> ```

To:

```<cls> mardkdown <sep> code <sep> code <sep> ... <sep> code <sep> ```

### Preprocessing
To prepare the data and extract features for training, including the markdown-only dataframes and sampling the code cells needed for each note book, simply run:

```$ python preprocess.py```

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

### Inference
- Add
