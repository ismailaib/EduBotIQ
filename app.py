from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json
import random

user_id = 2

app = Flask(__name__)

model = tf.keras.models.load_model('education_classifier_model')

tokenizer = Tokenizer()
data = pd.read_csv('training_data.csv')
texts = data['text'].tolist()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

with open('labels_mapping.json', 'r') as json_file:
    label_to_index = json.load(json_file)

responses = {}
responses_data = pd.read_csv('responses.csv')
for index, row in responses_data.iterrows():
    category = row['tag']
    response = row['response']
    if category in responses:
        responses[category].append(response)
    else:
        responses[category] = [response]

max_sequence_length = 10

def classify_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    prediction = model.predict(data)

    label_index = prediction.argmax()
    predicted_category = [key for key, value in label_to_index.items() if value == label_index][0]

    response = random.choice(responses.get(predicted_category, ["I'm not sure how to respond to that."]))

    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    predicted_response = classify_text(text)
    return jsonify({'result': predicted_response})

if __name__ == '__main__':
    app.run(debug=True)
