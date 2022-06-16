# ai4code-transformer-baseline

Solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition.

## Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and Khoi Nguyen's [baseline](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells) with numerous modifications. 

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

To change the number of feature samples:

```$ python preprocess.py --num_samples 20```

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

Output will be in the ```./data``` folder:

### Training Time

~ 20 Hours - Tesla A100-SXM4-40GB - 5 Epochs

~ 33 Hours - Tesla V100-SXM2-16GB - 5 Epochs

## Working Example

Preprocessing and Training Notebook: Add

Inference Notebook: https://www.kaggle.com/conanbranch/ai4code-transformer-baseline-inference/

## Results

Evalution on 20% of data where training set = 18% and the validation set = 2%:

| Batch Size |  Val. |
| --- | --- | 
| 8 | | 
| 16 | |    
| 32 | |   
| 64 | |  

| Epochs |  Val. |
| --- | --- | 
| 3 | | 
| 5 | |    
| 7 | |   
| 10 | |  

| lr |  Val. |
| --- | --- | 
| 1e-5 | | 
| 3e-5 | |    
| 5e-5 | | 

| weight <br> decay |  Val. |
| --- | --- | 
| .1 | | 
| .01 | |    
| .001 | |  

| activation |  Val. |
| --- | --- | 
| sigmoid | | 
| linear | | 

| eval |  Val. |
| --- | --- | 
| MSE | | 
| BSE | |

| eval |  Val. |
| --- | --- | 
| <SEP> | | 
| <EOS> | |
| No <SEP> | |

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 20 | | |
| 40 | | |   
| 60 | | |  
| 80 | | |  

| Code Cells <br> Sampled | MDL 32 <br> Val. | MDL 64 <br> Val. | MDL 128 <br> Val. |
| --- | --- | --- | --- |
| 20 |  |  |  | 
| 40 |  |  |  |  
| 60 |  |  |  |  
| 80 |  |  |  |    

| # Re-Init | Val. | BC Val. |
| --- | --- | --- | 
| 0 |  |  |  
| 1 |  |  | 
| 2 |  |  |   
| 4 |  |  |    
| 8 |  |  | 

Evalution on 10% of data where training set = 9% and the validation set = 1%

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 10 | .7921 | .7942 |
| 20 | .8066 | .8077 |
| 30 | **.8148** | .8177 | 
| 40 | .8124 | .8149 |   
| 50 | .8140 | .8154 |  
| 60 | .8138 | .8188 |  
| 70 | .8098 | .8098 |  
| 80 | .8045 | .8027 |  
| 90 | .7956 | .7995 |	
| 100 | .7937 |.8022 | 

| Code Cells <br> Sampled | MDL 32 <br> Val. | MDL 64 <br> Val. | MDL 128 <br> Val. |
| --- | --- | --- | --- |
| 10 | .7872 | .7921 |  |
| 20 | .8000 |.8066 |  | 
| 30 | .8089 |**.8148** |  | 
| 40 | **.8107** |.8124 |  |  
| 50 | .8040 |.8140 |  | 
| 60 | .8062 |.8138 |  |  
| 70 | .8050 |.8098 |  |  
| 80 | |.8045 |  |  
| 90 | |.7956 |  |  
| 100 | |.7937 |  | 

| # Re-Init | Val. | BC Val. |
| --- | --- | --- |
| 0 | **.8094** | .8060 | 
| 1 | .8069 | **.8084** |  
| 2 | .8077 | .8058  | 
| 3 | .8073 | .8066 |  
| 4 | .8052 | .8055 | 
| 5 | .8019 | .8040 | 
| 6 | .8046 | .8025 |
| 7 |  | .7970 |
| 8 |  | .7907 |
| 9 |  | .7849 |

## To Do (Code)

- Clean up ranking
- Explore https://github.com/huggingface/evaluate (includes function for average word length)

## To Do (Features)
- Add Stochastic Weight Averaging (SWA)
- Consider adding frequent evaluation
- Clean up input (remove comments from code, remove markup and other stuff from comments) Remove \r and \n from markup 
- Add optional no seperators [SEP]
- Add optional [EOS]

## To Do (Experiments)
- Test Variable Length Code
- Try end of sentence token [EOS]
- Try no seperators [SEP]
- Try adding activation function (sigmoid, will need to adjust ranking a tiny bit)
- Try MSE instead of BCE
- Sweep batch size (depending on GPU availailibity 8, 16 (current) with V100 or lower, 32, 64 with A100)
- Sweep learning rate (1e-5, 3e-5 (current), 5e-5) 
- Sweep weight decay (0.1, 0.01 (current), 0.001)
- Sweep epochs (3, 5(current), 7, 10)
- Sweep md_max_len (32, 64 (current), 128)
- Do any parents cross over?
