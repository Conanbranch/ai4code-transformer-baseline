# ai4code-transformer-baseline

Solution for [Google AI4Code](https://www.kaggle.com/competitions/AI4Code) competition.

## Overview
Based on Amet Erdem's [baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline) and Khoi Nguyen's [baseline](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells) with several modifications to the model and preprocessing. 

Code cells are sampled from the notebook to provide context for the markdown. The code cells are sampled uniformaly, perserving the order. Unlike Khoi Nguyen's baseline the their is no <SEP> and <PAD> tokens between the sampled code. Input will look like this: 

```<CLS> markdown <SEP> code_1, code_2, ..., code_n <SEP>```

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

### Variable Length Code

To use variable length code:

```$ python train.py --md_max_len 64 --total_max_len 512 --batch_size 16 --accumulation_steps 4 --epochs 5 --n_workers 8 --vbl_code True```

The default is ```code_max_length = int((self.total_max_len - self.md_max_len)/num_samples)```, where the code size (number of tokens per code cell) is fixed regardless of the number of code cells in the notebook. With variable length code, ```code_max_length = int((self.total_max_len - self.md_max_len)/num_sampled)``` and is a function of the actual number of code cells sampled (i.e. there were only 7 code cells ```num_sampled = 7``` but ```num_samples = 20```, this gives you more tokens per code cells when ```num_sampled < num_samples```).

### Training Time

~ 20 Hours - Tesla A100-SXM4-40GB - 5 Epochs - 54.8

~ 33 Hours - Tesla V100-SXM2-16GB - 5 Epochs - 54.8 Gb RAM

## Working Example

Preprocessing and Training Notebook: Add

Inference Notebook: https://www.kaggle.com/conanbranch/ai4code-transformer-baseline-inference/

## Results - 100% Dataset Eval

| epochs | lr | md max | features | re-init | public lb
| --- | --- | ---|  --- | ---| --- |
| 10 | 3e-5 | 64 | 40 | 1 |  |

## Results - 90% Dataset Eval

| epochs | lr | md max | features | re-init | val. | public lb
| --- | --- | ---|  --- | ---|  --- | --- |
| 5 | 5e-5 | 64 | 40 | 1 | .8589 | .8526 |
| 10 | 3e-5 | 64 | 30 | 1 |	|  ||
| 10 | 3e-5 | 64 | 40 | 0 | 	|  |
| 10 | 2e-5 | 64 | 40 or 30 | 1 | 	|  |
| 10 | 3e-5 | 64 | 40 | 1 | **.8650**	| **.8575** |
| 10 | 3e-5 | 64 | 40 | 2 | .8628	| .8548 |
| 10 | 3e-5 | 64 | 40 | 3 | .8626 | .8558 |
| 10 | 3e-5 | 64 | 40 | 4 |  | |
| 10 | 4e-5 | 64 | 40 | 1 | .8641	| .8567 |
| 10 | 5e-5 | 64 | 40 | 1 | .8639	| .8558 |

## Results - 20% Dataset Eval

90/10 split evalution on 20% of data where training set = 18% and the validation set = 2%:

<details>
  <summary><b>Initial Evaluation (Mostly Linear Activation)</b></summary>
  &nbsp;
  
| Code Cells <br> Sampled | MDL 64 <br> Val. | MDL 128 <br> Val. |
| --- | --- | --- | 
| 20 | .8323 | .8326  | 
| 40 | **.8385** | **.8394** |  
| 60 | .8377 | .8357 |  
| 80 | .8302 | .8248 |

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 20 | .8323 | .8318 |
| 40 | **.8385** | .8420 |   
| 60 | .8377 | **.8426**  |  
| 80 | .8302 | .8409 |  

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
| 15 | .8424 |  
| 20 | **.8445** |  

Very little gain from 10, but double the training time

| lr |  Val. |
| --- | --- | 
| 1e-5 | .8309 | 
| 3e-5 | .8385 |    
| 5e-5 | **.8428** | 

Does not help with larger training sizes. Performance appears to be a function of learning rate, epochs, and training set size

| weight <br> decay |  Val. |
| --- | --- | 
| .1 | **.8389** | 
| .01 | .8385 |    
| .001 | .8373 |  

| activation | BCE Val. | 
| --- | --- | 
| tanh | .8401 |
| sigmoid | **.8438** | 
| linear | .8385|

| # Re-Init | Val. | 
| --- | --- | 
| 0 | .8385 | 
| 1 | .8400 | 
| 2 | .8392 |   
| 4 | **.8401** |    
| 8 | .8206 | 

| eval |  FLC Val.\** | 
| --- | --- | 
| Code \<SEP\> | .8385 | 
| No Code \<SEP\> | .8372 | 
| No Code \<PAD\>* | .8391 | 
| No Code \<SEP\> and \<PAD\>* | **.8397**\** |

\*No pad between code, just after

\*\*Not sure if their is a real change
  
</details>  
 
<details>
  <summary><b>Sigmoid Activation</b></summary>
  &nbsp;

64 MD Size

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 30 | **.8424** | ***.8430*** |
| 35 | .8387 | .8384 |
| 40 | **.8405** | *.8429* |
| 45 | .8401 | ***.8432*** |
| 50 |  .8385  | *.8389* |  

128 MD Size

| Code Cells <br> Sampled | FLC <br> Val. | VLC <br> Val. |
| --- | --- | --- |
| 30 | **.8418** | |
| 35 | .8377 | |
| 40 | .8390 | |
| 45 | **.8420** | |
| 50 | .8385 | |  

LWRD (test at 10 epochs re-init 1)

| head lr | tail lr | Val. |
| --- | --- | --- |
| 2e-5 | 1e-5 | |
| 3e-5 | 1e-5 | |
| 3e-5 | 2e-5 | |
| 4e-5 | 1e-5 | |
| 4e-5 | 2e-5 | |
| 4e-5 | 3e-5 | |
| 5e-5 | 1e-5 | |
| 5e-5 | 2e-5 | |
| 5e-5 | 3e-5 | |
| 5e-5 | 4e-5 | |

| eval (code) |  Val.* |
| --- | --- | 
| Default | .8424 | 
| All |	.8396 | 
| Newlines |	**.8432**\* |
| Lower | .8421 |
| Tokens | .8430 |
| Comments | .8421 |

\*Does not appear to be any real change from default

| eval (md) |  Val.\* |
| --- | --- | 
| Default | .8424 | 
| All | .8361 | 
| Markdown|	.8384 |
| Special Characters | .8398 |
| Special Characters Except | .8423 |
| Tokens | .8418 |
| Lowercase | **.8436**\* |
| Extra | .8366 |
| URL | .8419 |
| Newlines |  |

\*Does not appear to be any real change from default

| output | Val. |
| --- | --- | 
| lhs | **.8438** |
| mean pooling | .8394 |
| max pooling |  |
| mean max pooling |  |

| experimemt | Val. |
| --- | --- | 
| newlines | .8432 |
| newlines no sep and pad | **.8443** |
| no sep and pad | .8431 |
| normal | .8424 |
  
Tested at 10 Epochs re-init 1:

| weight <br> decay |  Val. |
| --- | --- |
| .2 | | 
| .1 | | 
| .01 | |    

| lr |  Val. |
| --- | --- | 
| 2e-5 | | 
| 3e-5 | |    
| 4e-5 | | 

| BS |  Val. |
| --- | --- | 
| 8 | | 
| 16 | |    
  
</details>    

## Full Dataset Eval
  
<details>
  <summary><b>To Do</b></summary>
  &nbsp;  

Best Epochs:

9,10,11

Batch Size:

8, 16

LR:

1e-5, 2e-5, 2.9e-5, 3e-5
  
  </details>

## To Do (Code)
  
  <details>
  <summary><b>Code</b></summary>
 
  - Clean up ranking
  </details> 
  
  <details>
  <summary><b>Experiments</b></summary>
    
  - Try training the 90% for more epochs
  - Might be worth probbing leaderboard with vlc 40 and vlc 50 to see if 50 does better on leaderboard
  - Do evaluation with different rankings, revert if no change 
    
  </details> 
  
  <details>
  <summary><b>Final Model</b></summary>
  
  - Train final models on 90% of data 
 
  </details> 
  
  <details
   
  <details>
  <summary><b>Features (did not get to)</b></summary>

  - Adversarial Weight Perturbation (AWP)
  - Frequent evaluation 
  - Stochastic Weight Averaging (SWA) 
  
  </details> 
