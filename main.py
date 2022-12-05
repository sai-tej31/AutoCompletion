import pandas as pd 
import numpy as np 
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense

from train import build_model,train_model,predict

### Data Preprocess

df = pd.read_csv('all-data.csv', encoding='latin-1',header=None)

def preprocess_data(df):
    textual_data = df[1]
    data = [i.strip() for i in textual_data]
    data = [re.sub("[^a-zA-Z0-9 ]","",i) for i in data]
    return data

    

### Create Unique Word Dictionary
def create_words(df):
    data = preprocess_data(df)
    words = []
    for sent in data:
        for word in sent.split(' '):
            if word and word not in words:
                words.append(word)
    return words

words = create_words(df)

####Get the words from main python file

tokenizer = Tokenizer()
tokenizer.fit_on_texts([words])

sequence_data = tokenizer.texts_to_sequences([words])[0]

bigram_and_output = []

for i in range(2, len(words)):
    x_words = sequence_data[i-2:i+1]
    bigram_and_output.append(x_words)

##Create Input and output arrays
X,Y = [],[]
for i in bigram_and_output:
    X.append(i[0:2])
    Y.append(i[2])

X= np.array(X)
Y = np.array(Y)

Y = to_categorical(Y,num_classes=len(tokenizer.word_index)+1)

model = train_model(tokenizer)

input1 = input('Enter Here:')
print(predict(model, tokenizer, input1))




