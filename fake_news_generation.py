# -*- coding: utf-8 -*-
"""Fake_News_Generation.ipynb

# Fake News Generation

project exploring how neural networks can be used to create a language model that can generate text and learn the rules of grammar and English :) well, by applying the knowledge for evil and learn how to generate fake news.
"""

#@title import libraries and download the data. If there is a prompt, just enter "A"
import os
import random
import string
import sys
from collections import Counter
from ipywidgets import interact, interactive, fixed, interact_manual

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown
import warnings
warnings.filterwarnings('ignore')
gdown.download("https://drive.google.com/uc?id=11WClewW80aEj8RrdmS9qkchwQsOkJlHy", 'fake.txt', True)
gdown.download("https://drive.google.com/uc?id=1UuANHblVzkclCC2v9J0V7uxX0Y0Fjfkx", 'pre_train.zip', True)

! unzip -oq pre_train.zip

#@title load helper functions
def load_data():
    with open("fake.txt", "r") as f:
        return f.read()


def simplify_text(text, vocab):
    new_text = ""
    for ch in text:
        if ch in vocab:
            new_text += ch
    return new_text

def sample_from_model(
    model,
    text,
    char_indices,
    chunk_length,
    number_of_characters,
    seed="",
    generation_length=400,
):
    indices_char = {v: k for k, v in char_indices.items()}
    for diversity in [0.2, 0.5, 0.7]:
        print("----- diversity:", diversity)
        generated = ""
        if not seed:
            text = text.lower()
            start_index = random.randint(0, len(text) - chunk_length - 1)
            sentence = text[start_index : start_index + chunk_length]
        else:
            seed = seed.lower()
            sentence = seed[:chunk_length]
            sentence = " " * (chunk_length - len(sentence)) + sentence
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for _ in range(generation_length):
            x_pred = np.zeros((1, chunk_length, number_of_characters))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print("\n")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64") + 1e-8
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class SampleAtEpoch(tf.keras.callbacks.Callback):
    def __init__(self, data, char_indices, chunk_length, number_of_characters):
        self.data = data
        self.char_indices = char_indices
        self.chunk_length = chunk_length
        self.number_of_characters = number_of_characters
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        sample_from_model(
            self.model,
            self.data,
            self.char_indices,
            self.chunk_length,
            self.number_of_characters,
            generation_length=200,
        )


def predict_str(model, text, char2indices, top=10):
    if text == '':
      print("waiting...")
      return
    text = text.lower()
    assert len(text) < CHUNK_LENGTH
    oh = np.array([one_hot_sentence(text, char2indices)])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      pred = model.predict(oh).flatten()
    sort_indices = np.argsort(pred)[::-1][:top]
    plt.bar(range(top), pred[sort_indices], tick_label=np.array(list(VOCAB))[sort_indices])
    plt.title(f"Predicted probabilities of the character following '{text}'")
    plt.show()

"""## Language models

language model will look at the previous words in a sequence and use that compute the probabilities of what the next word will be. 

The next cell defines some constants that will be used in the language model

*   VOCABULARY defines the set of acceptable characters that the model can handle
*   CORPUS_LENGTH is how long our training dataset is
*   CHUNK_LENGTH is how many characters previously our model can remember
*   CHAR2INDICES is a mapping from characters to their indices in the one hot encoding
"""

STEP = 3
LEARNING_RATE = 0.0005
CORPUS_LENGTH = 200000
CHUNK_LENGTH = 40
VOCAB = string.ascii_lowercase + string.punctuation + string.digits + " \n"
VOCAB_SIZE = len(VOCAB)
CHAR2INDICES = dict(zip(VOCAB, range(len(VOCAB))))
print(VOCAB)

"""load data and simplify the text a bit by removing all the characters that are not in our vocabulary.
note dataset is a sequence of fake news articles all compiled to one long string"""

data = load_data()
data = data[:CORPUS_LENGTH]
data = simplify_text(data, CHAR2INDICES)
print(f"Type of the data is: {type(data)}\n")
print(f"Length of the data is: {len(data)}\n")
print(f"The first couple of sentence of the data are:\n")
print(data[:500])

"""## Encoding words
"""

def one_hot(char, char_indices):
    num_chars = len(char_indices)
    vec = [0] * num_chars # Start off with a vector of all 0s
    ### BEGIN YOUR CODE ###
    vec[char_indices[char]] = 1
    ### END YOUR CODE ###
    return vec


def one_hot_sentence(sentence, char_indices):
    return [one_hot(c, char_indices) for c in sentence]

"""When you've got it, test it below, try typing 'abc', and see if you get what you would expect!"""

interact(lambda text: np.array(one_hot_sentence(text, CHAR2INDICES)), text="a");

#@title Run this to load a helper function :)
def get_x_y(text, char_indices):
    """
    Extracts X and y from the raw text.
    
    Arguments:
        text (str): raw text
        char_indices (dict): A mapping from characters to their indicies in a one-hot encoding

    Returns:
        x (np.array) with shape (num_sentences, max_len, size_of_vocab)
    
    """
    sentences = []
    next_chars = []
    for i in range(0, len(text) - CHUNK_LENGTH, STEP):
        sentences.append(text[i : i + CHUNK_LENGTH])
        next_chars.append(text[i + CHUNK_LENGTH])

    print("Chunk length:", CHUNK_LENGTH)
    print("Number of chunks:", len(sentences))

    x = []
    y = []
    for i, sentence in enumerate(sentences):
        x.append(one_hot_sentence(sentence, char_indices))
        y.append(one_hot(next_chars[i], char_indices))

    return np.array(x, dtype=bool), np.array(y, dtype=bool)

"""convert raw fake new articles into arrays that can be used in our model. """

print("This might take a while...")
x, y = get_x_y(data, CHAR2INDICES)
print("Shape of x is", x.shape)
print("Shape of y is ", y.shape)

"""## Building the Language Model

LSTM language model specializes in sequences. 

Tensorflow and Keras provides a implementation for LSTMs. 

The sequential model has two layers, the first layer is an LSTM layer, and the second layer should be a Dense layer.

The first layer (LSTM)
* should have 100 units
* should not return sequences
* should have input_shape (chunk_length, number_of_characters).

The Dense layer 
* should have number_of_characters neurons
* should have softmax activation, 

helpful documentation [here](https://keras.io/layers/recurrent/)
"""

def get_model(chunk_length, number_of_characters, lr):
    model = tf.keras.Sequential()
    ### YOUR CODE HERE
    model.add(
        tf.keras.layers.LSTM(
            100,
            return_sequences=False,
            input_shape=(chunk_length, number_of_characters),
        )
    )
    model.add(tf.keras.layers.Dense(number_of_characters, activation="softmax"))
    ### END CODE

    optimizer = tf.keras.optimizers.RMSprop(lr=lr)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model

model = get_model(CHUNK_LENGTH, VOCAB_SIZE, LEARNING_RATE)
model.summary()

"""# Fitting the model 
calling the fit function. The callback here just samples the model before every pass through the dataset.

### try running the model

Run the model for 3 epochs.
"""

sample_callback = SampleAtEpoch(data, CHAR2INDICES, CHUNK_LENGTH, VOCAB_SIZE)

model.fit(
    x, y, callbacks=[sample_callback], epochs=3,
)

model = tf.keras.models.load_model("cp.ckpt/")

SEED = "the government"
sample_from_model(model, data, CHAR2INDICES, CHUNK_LENGTH, VOCAB_SIZE, seed=SEED)

"""## What has our model learned? 

From the generated samples, it has started to learn some important details about the English language. Surely a huge improvement over the random gibberish from the start. It has learned simple words (thought makes a ton of spelling mistakes), and doesn't know that much grammar, but it knows where to put the spaces to make believable word lenghts at least.


*   Has it learned that the letter that follows 'q' is usually a 'u'?
*   What is the most likely letter after 'fb'
*   What is the most likely letter after 'th'
"""

interact(lambda sequence: predict_str(model, sequence, CHAR2INDICES), sequence='th');

"""## More things to try

"""

