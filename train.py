from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
import numpy as np


def build_model(tokenizer):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1,100,input_length=2))
    model.add(LSTM(2000,return_sequences=True))
    model.add(LSTM(2000))
    model.add(Dense(2000,activation="relu"))
    model.add(Dense(len(tokenizer.word_index)+1,activation="softmax"))
    return model

def train_model(tokenizer):
    model = build_model(tokenizer)
    model.compile(loss="categorical_crossentropy",optimizer="Adam")
    model.fit(X,Y,epochs=10,batch_size=256)
    return model

def predict(model, tokenizer, input1):
    sequence = tokenizer.texts_to_sequences([input1.split(" ")[-2:]])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""
  
    for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
    return predicted_word