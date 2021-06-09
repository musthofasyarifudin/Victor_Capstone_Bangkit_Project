# Capstone Bangkit Project 2021

## We Made Application that detect person disease with symptom based prediction
our team craeted android application that predict person disease

## Our Model
we create several model with each it own distinction. we eventually use lstm neural net for our api. to start using you can go to folder in each model and run notebook to try and see for yourself.

1. XGBOOST - Acc = 1.00
2. Tensorflow Decission Tree - Acc = 1.00
3. Tensorflow LSTM Neural Net - Acc = 0.98
4. Tensorflow Categorical Embedding LSTM Neural Net -Acc = 1

## API
We made api using flask framework you can go straight to API folder. In tHere there is main file and model file and class object. to start using prediction you can start with main.py and run. after that you can send json using postman to local address and it will send back json with prediction result.

## Android
We moved android repo in this link https://github.com/arissetyawan1/ui_victor_capstone. you can download and run in your android studio.
