from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from autocorrect import Speller
spell = Speller(lang='en')
import contractions
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
stop = stopwords.words('english')
import numpy as np
from nltk.tokenize import word_tokenize

# load the model from disk
model = keras.models.load_model('model_lstm_text_input_finalized_keras.h5')
#model=pickle.load(open('model_lstm_text_input.pkl','rb'))
#text_preprocessing=pickle.load(open('text.pkl','rb'))



def preprocess(userText):
    userText=userText.lower()
    #print(userText)
    userText=contractions.fix(userText)
    #print(userText)
    userText=" ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",userText) if s])
    #print(userText)
    userText=spell(userText)
    #print(userText)
    userText=re.sub('\W+',' ',userText)
    #print(userText)
    userText=userText.strip()
    #print(userText)
    userText=re.sub('[^A-Za-z0-9]+', ' ', userText)
    #print(userText)
    stop_words = set(stopwords.words('english'))
    words=word_tokenize(userText)
    #print(words)
    words_new=[i for i in words if i not in stop_words]
    #print(" ".join(words_new))
    return " ".join(words_new)
    	

def get_tokenizer(text):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    #print(tokenizer.texts_to_sequences(text))
    return tokenizer.texts_to_sequences(text)


maxlen = 99
def get_pad_sequences(text):
    return pad_sequences(text, padding='post', maxlen=maxlen)
	

def predict_class(text_pad, model):
    #print(model.predict(text_pad))
    return model.predict(text_pad)

	

def chatbot_response(text):
    text_tokenizer = get_tokenizer(text)
    #print(text_tokenizer)
    text_pad = get_pad_sequences(text_tokenizer)
    #print(text_pad)
    respond = predict_class(text_pad, model)
    print(np.argmax(respond))
    return np.argmax(respond)


app = Flask(__name__)
app.static_folder = 'static'

@app.route('/', methods=['GET', 'POST'])
def index(): 
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    text = preprocess(userText)
    #my_prediction = clf.predict(userText)
    result = str(chatbot_response(text))
    print(result)
    return result
     
   

if __name__ == "__main__": 
    app.run(debug=True)