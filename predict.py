from train import build_model,train_model,predict
import pickle
from tensorflow import keras
model = keras.models.load_model('model_main.h5')

tokenizer = pickle.load(open('token.pkl','rb'))
input1 = input('Enter Here:')
print("The Output: ",predict(model, tokenizer, input1))