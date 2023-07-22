from flask import Flask, render_template, request
from transformers import BertTokenizer
import torch
import torch.nn as nn
import model
import re

app = Flask(__name__)

model = model.BertRegressor(drop_rate=0.2)
model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')), strict=False)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data_for_BERT(text):
    text = text.lower()
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\r\n\t', ' ', text)
    text = re.sub(r'[^\w\s!\\/]', '', text)

    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    text = re.sub(pattern, '', text)
    return text

def encode(text):
    encoder =  tokenizer(text=text,
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=512,
                            return_attention_mask=True)
    return torch.tensor(encoder['input_ids'])[None, :], torch.tensor(encoder['attention_mask'])[None, :]

def transform(x):
    x = int(round(x))
    if x > 10:
        return 10
    if x < 1:
        return 1
    return x

def get_sentiment(value):
    if value >=5:
        return 'positive'
    return 'negative'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form['review_text']
    ids, mask = encode(preprocess_data_for_BERT(text))
    output = torch.flatten(model(ids, mask)).tolist()[0]
    output = transform(output)
    sentiment = get_sentiment(output)
    return render_template("index.html", prediction_text = "Rating: {0}, Sentiment: {1}".format(output, sentiment))


if __name__ == '__main__':
    app.run()
