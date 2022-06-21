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

| Code Cells <br> Sampled | MDL 32 <br> Val. | MDL 64 <br> Val. | MDL 128 <br> Val. |
| --- | --- | --- | --- |
| 20 |  | .8323 | .8326  | 
| 40 |  | **.8385** | **.8394** |  
| 60 |  | .8377 | .8357 |  
| 80 |  | .8302 | .8248 |

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 20 | .8323 | |
| 40 | **.8385** | |   
| 60 | .8377 | |  
| 80 | .8302 | |  

Note: If VLC does better, consider testing re-init with it as well as the different MDL

Aside from the above, all below experiments are a sample of 40, a linear activation, and an MDL of 64

| Batch Size* |  Val. |
| --- | --- | 
| 8 | **.8385** | 
| 16 | **.8385** |    
| 32 | .8359 |   
| 64 | .8353 |  

\*depending on GPU availailibity 8, 16 (current) with V100 or lower, 32, 64 with A100

| Epochs |  Val. |
| --- | --- | 
| 3 | .8363 | 
| 5 | .8385 |    
| 7 | .8411 |   
| 10 | .8437 |  
| 15 |  |  
| 20 |  |  

| lr |  Val. |
| --- | --- | 
| 1e-5 | | 
| 3e-5 | .8385 |    
| 5e-5 | | 

| weight <br> decay |  Val. |
| --- | --- | 
| .1 | | 
| .01 | .8385 |    
| .001 | |  

| activation | BCE Val. | MSE Val. |
| --- | --- | ---|
| tanh | .8421 | |
| sigmoid | **.8438** | | 
| linear | .8385| |

| # Re-Init | Val. | 
| --- | --- | 
| 0 | .8385 | 
| 1 | .8400 | 
| 2 | .8392 |   
| 4 | **.8401** |    
| 8 | .8206 | 

| eval |  FLC Val. |  VLC Val. |
| --- | --- | --- |
| Code \<SEP\> | .8385 | |
| No Code \<SEP\> | | |
| No Code \<PAD\>* | | |
| No Code \<SEP\> and \<PAD\>* | | |
\*No pad between code, just after

| eval |  Val. |
| --- | --- | 
| Code (//n) | | 
| Markdown | |

## Sigmoid Eval

| # Re-Init | Val. | 
| --- | --- | 
| 0 |  | 
| 1 |  | 
| 2 |  |   
| 4 |  |    
| 8 |  | 

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 20 |  | |
| 30 |  | |
| 40 |  | |
| 50 |  | |  
| 60 |  | |  
| 80 |  | |  

## Full Dataset Eval

Re-Init = 1
Code Sampled = 40
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 Takes ~ 7 Days)

Re-Init = 1
Code Sampled = 30
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 Takes ~ 7 Days)

Re-Init = 2
Code Sampled = 40
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 Takes ~ 7 Days)

Re-Init = 2
Code Sampled = 30
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 Takes ~ 7 Days)

Re-Init = 3
Code Sampled = 40
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 Takes ~ 7 Days)

Re-Init = 3
Code Sampled = 30
MDL = 64
Activation = Sigmoid
Epochs = 10 ~ 3 Days (25 ~ 7 Days)

## To Do (Code)
- Clean up ranking

## To Do (Features)

- Frequent evaluation - May not make any difference, don't bother unless really need a boost
- Add Stochastic Weight Averaging (SWA) - Complicated, Final Step Only If Necessary - Some chatter that it doens't help anyway, don't bother unless really need a small boost

## To Do (Experiments)
- Try no padding except at end of markdown and code
- Maybe explore https://github.com/huggingface/evaluate (includes function for average word length)
- Different pooling strategies https://www.kaggle.com/code/conanbranch/utilizing-transformer-representations-efficiently/edit
- Might be able to include comments in with the code for the context, even if not in the correct order. Maybe.
- Try sigmoid with proper scalling where all ranks are between 0 - 1

## To Try (Input Preprocessing)

Note: Heavy processing may not be a good idea as transformers can utilize context.

- Clean up input (remove comments from code, remove markup and other stuff from comments) Remove \r and \n from markup 
- https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
- https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-training/notebook
