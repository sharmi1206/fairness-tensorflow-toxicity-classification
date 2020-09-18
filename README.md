# fairness-tensorflow-toxicity-classification
Tensorflow constrained optimization with different Deep Learning Models 

The code repository contains intgeration with tensorflow-constrained-optimization for introducing fairness in ML models. Models used here are

1. Simple LSTM
2. CNN
3. Bi-Directional LSTM 

All these models are available at notebook Wiki_toxicity_fairness-lstm-cnn-bi-lstm.ipynb

It also contains

4. Stacked LSTM and CNN
5. Stacked BI-LSTM and CNN

These models are available at Wiki_toxicity_fairness-stacked-lstm-cnn.ipynb

It further contains integration with

6. BERT in toxicity_classification_fainess_bert.ipynb  

Note : Since creating bert embeddings and tokenizing them takes time, we have limited training to 5000 data points. Of course this can ve extended to full dataset.


How to Download the data?

1. Ensure you have a folder called fair_data in inside the main repo
2. Create a sub-folder jigsaw-unintended-bias-in-toxicity-classification 
3. Download data from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data . In the scope of this POC, we limit data training only from train.csv ie. 50% for training and remaining 50% for training and validation, yielding to 902437 training points and 451219 points for each testing and validation. 


For Bert models:

Ensure you are downloading the model and the following lines are uncommneted.

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
os.makedirs("model", exist_ok=True)
!mv uncased_L-12_H-768_A-12/ model

Ensure you are training the bert model using:

model = multi_cls_create_model(max_seq_len=128, bert_ckpt_file). 

If model is set to  something else, reset to bert model


