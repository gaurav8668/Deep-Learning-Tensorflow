import json
import numpy as np 
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama     #  cross-platform API to print colored terminal text from Python applications.
colorama.init()  # If you are on Windows, the call to ‘’init()’’ will start filtering ANSI escape sequences out of any text 
from colorama import Fore, Style, Back

import random
import pickle

with open(r'C:\Users\jgaur\Tensorflow_Tut\chatbot\data.json') as file:
    data = json.load(file)
    
def chat():
    # load trained model
    model = keras.models.load_model(r'C:\Users\jgaur\Tensorflow_Tut\chatbot\chat_model')

    # load tokenizer object
    with open(r'C:\Users\jgaur\Tensorflow_Tut\chatbot\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # load encoder object
    with open(r'C:\Users\jgaur\Tensorflow_Tut\chatbot\label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: ", Style.RESET_ALL, end="")
        inp = input()
        if inp == 'quit' or inp == "QUIT":
            break
        
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), 
                                                        truncating='post', maxlen=max_len))
                                                    
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot: ", Style.RESET_ALL, np.random.choice(i['responses']))
            
print(Fore.YELLOW + "Start messaging with the bot (type 'quit' or 'QUIT' to stop)!" + Style.RESET_ALL)
chat()