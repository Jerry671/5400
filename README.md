# 5400 Group Project
# What does this code do?
Our system employs deep learning to model and classify emotions in poems using large-scale annotated poetry data sets.

# How to run
Our code has the application programming interface. Enter this code in the command line:

poem_sentiment_analysis % python3poem_emotion_predict.py

(base) bzb@haobozhideMacBook-Pro poem_sentiment_analysis % python3 poem_emotion_predict.py
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://753f857b2e8a3344ff.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
to be or not to be, that is a question
sad
床前明月光
miss

# Project structure
folder data: save all the data we need for models
folder new data: save all the data for testing the model
folder flagged: save all the results that flagged by user into json files
folder models: save all the model.pkl files that are needed
dataset.py: defines a PyTorch dataset for sentiment analysis using our dataset, with Chinese text processing.
build_vocab.py: defines a vocabulary class, processes datasets for sentiment analysis, builds a vocabulary, and saves it using pickle.
vocab.py: defines a vocabulary class (Vocab) for text serialization, which includes methods for fitting, building a vocabulary, and transforming text sequences.
split_train_test.py:defines a function split_data() to split a dataset into training and testing sets based on specified conditions for each emotion label.
tang_sum_emotion.py: predicts emotions for a set of poems using a pre-trained BiLSTM model with Attention and analyzes the overall emotion distribution.
test.py: test the performance of the model with part of the data
poem_emotion_attention.py: leverage the attention mechanism
poem_attention_bilstm_model.py:leverage the BiLSTM model
poem_emotion_predict.py: defines a sentiment analysis model for Chinese poems using an BiLSTM with Attention and creates a Gradio interface for predictions.

* The github repo for this project: https://github.com/Jerry671/5400
* The share link of our user interface expires in 72 hours. If you want to try the user interface, please email us to get the link: bh791@georgetown.edu
