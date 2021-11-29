# A BERT-CRF-Based Sequence Tagging in Pytorch Implementation
----
## Quick Start
1. Change Config/config.json  
2. if this is your first time run the project, run `python preprocess.py` before training   
2. to train your model, run `python train.py`  
3. to eval test set, run `python test.py`  
4. if figures of records is needed, run `python figure_painter.py`

## Some Instructions
1. The backbone of the model is BERT-BASE.   
I add the **batch-computable CRF** layer after BERT. The source code is borrowed from [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html). 

2. Also, the CRF layer is removable.

3. Each model will be tagged a *time\_flag* at the beginning of training. Thus, the checkpoints, logs, evaluation results, as well as train data records can be distinguished by the *time\_flag* of their file name or dir name. 

4. Historically optimal checkpoint will be stored into `./checkpoints/`.  

5. Log file in `./log/` saves every item of the eval scores of val set during training.

6. Evaluation scores on test set is stored into `./results/`.

7. For ploting training record into figures, all scores are saved as a .pkl file and will be stored into `./training_data/`.

----
----


