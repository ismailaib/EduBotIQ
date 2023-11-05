from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('education_classifier_model_v2')

# Load the tokenizer and word index
tokenizer = Tokenizer()
data = pd.read_csv('training_data.csv')
texts = data['text'].tolist()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Load the label-to-index mapping
with open('labels_mapping.json', 'r') as json_file:
    label_to_index = json.load(json_file)

# Define the maximum sequence length
max_sequence_length = 10

# Define a function to preprocess and classify the text
def classify_text(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Make a prediction using the model
    prediction = model.predict(data)

    # Map the prediction to a label in their original order
    label_index = prediction.argmax()
    predicted_label = [key for key, value in label_to_index.items() if value == label_index][0]

    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    predicted_label = classify_text(text)
    return jsonify({'result': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
