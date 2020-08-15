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


How to Download the data?

1. Ensure you have a folder called fair_data in inside the main repo
2. Create a sub-folder jigsaw-unintended-bias-in-toxicity-classification 
3. Download data from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data


For Bert models:

Ensure you are downloading the model and the following lines are uncommneted.

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
os.makedirs("model", exist_ok=True)
!mv uncased_L-12_H-768_A-12/ model
