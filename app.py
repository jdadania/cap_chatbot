import pickle

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
  my_prediction = clf.predict(userText)
  return my_prediction
   

if __name__ == "__main__":
    app.run()