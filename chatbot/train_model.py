import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # silence some warnings

import tensorflow as tf 
import json # Data representatin format, Lightweight and reasy to read/write
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

with open(r'C:\Users\jgaur\Tensorflow_Tut\chatbot\data.json') as file:
    data = json.load(file)

training_sentence = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentence.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

# ------------------------------- Label Encoding ----------------------------
# Encode the target labels with value between 0 and n_classes-1
lbl_encoder = LabelEncoder()
# print(training_labels)
lbl_encoder.fit(training_labels)
# print(training_labels)
training_labels = lbl_encoder.transform(training_labels)
# print(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20

#-------------------------------------- Tokenization ------------------------
# Vectorize the text corpus by turning eaach text into a sequence of integer
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(training_sentence)
word_index = tokenizer.word_index
# print(word_index)
training_sequence = tokenizer.texts_to_sequences(training_sentence)
# print(training_sequence)
padding_sequence = pad_sequences(training_sequence, truncating='post', maxlen=max_len)
# print(padding_sequence)

# -------------------------------------------  Training Model----------------------------------
model = Sequential()
# Feature representatin of a particular word
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())     # Scale the dimensions
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(padding_sequence, np.array(training_labels), epochs=500, verbose=2)

# ---------------------------------- Savting the model -------------------------------------
# Saving the model
model.save("chat_model")

# to save the fitted tokeniser
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)



print("code completed")