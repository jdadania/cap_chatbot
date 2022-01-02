import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import contractions
from autocorrect import Speller
spell = Speller(lang='en')
text_preprocessing=pickle.load(open('text.pkl','rb'))

def get_tokenizer(text):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    return tokenizer.texts_to_sequences(text)



maxlen = 100
def get_pad_sequences(text):
    return pad_sequences(text, padding='post', maxlen=maxlen)
	

def predict_class(text_pad, clf):
    return clf.predict(text_pad)

	

def chatbot_response(text):
    text_tokenizer = get_tokenizer(text)
    text_pad = get_pad_sequences(text_tokenizer)
	respond = predict_class(text_pad, clf)
    return respond
	
	
from flask import Flask, render_template, request

# load the model from disk
filename = 'model_lstm_text_input.pkl'
clf = pickle.load(open(filename, 'rb'))



app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
  userText = request.args.get('msg')
  text = text_preprocessing.text(userText)
  #my_prediction = clf.predict(userText)
  return chatbot_response(text)
   

if __name__ == "__main__":
    app.run()