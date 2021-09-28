import json
import nltk
from nltk.stem import WordNetLemmatizer
import string
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras import metrics
from tensorflow.python.keras.layers.core import Dense

class TrainingModel:
    def __init__(self, words, classes, data_x, data_y):
        self.words = words
        self.classes = classes
        self.data_x = data_x
        self.data_y = data_y
        self.lemmatizer = WordNetLemmatizer()

    def train(self):
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
        training = []
        out_empty = [0]* len(classes)

        for idx, doc in enumerate(data_x):
            bow = []
            text = lemmatizer.lemmatize(doc.lower())

        for word in words:
            bow.append(1) if word in text else bow.append(0)

        output_row = list(out_empty)
        output_row[classes.index(data_y[idx])] = 1

        training.append([bow, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = np.array(list(training[:,0]))
        train_y = np.array(list(training[:,1]))

        input_shape = (len(train_x[0]), )
        output_shape = len(train_y[0])

        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation="softmax"))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
            optimizer=adam,
            metrics=["accuracy"])

        model.fit(x=train_x, y=train_y, epochs=500, verbose=0)

        return model
    
    def clean_text(text):
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
        return tokens

    def bag_of_words(command, words):
        tokens = clean_text(command)
        bow = [0]* len(words)

        for token in tokens:
            for idx, word in enumerate(words):
                if word == token:
                    bow[idx] = 1

        return np.array(bow)

    def get_intent(command, words, classes):
        bow = bag_of_words(command, words)
        result = model.predict(np.array([bow]))[0]
        thresh = 0.5

        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)

        return_list =[]

        for pred in y_pred:
            return_list.append(classes[pred[0]])
        return return_list

    @staticmethod
    def get_response(intents, data):
        tag = intents[0]
        list_of_intents = data['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                response = random.choice(intent['response'])