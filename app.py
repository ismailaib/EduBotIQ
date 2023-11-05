from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('education_classifier_model')

# Load the tokenizer and word index
tokenizer = Tokenizer()
data = pd.read_csv('training_data.csv')
texts = data['text'].tolist()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Define the maximum sequence length
max_sequence_length = 10

# Define a function to preprocess and classify the text
def classify_text(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Make a prediction using the model
    prediction = model.predict(data)

    # Map the prediction to a label
    label_index = prediction.argmax()
    labels = ['greeting', 'goodbye', 'creator', 'name', 'hours', 'greeting', 'course', 'fees', 'location', 'hostel', 'event', 'document', 'floors', 'syllabus', 'library', 'infrastructure', 'canteen', 'menu', 'placement', 'ithod', 'computerhod', 'extchod', 'principal', 'sem', 'admission', 'scholarship', 'facilities', 'college intake', 'uniform', 'committee', 'random', 'swear', 'vacation', 'sports', 'salutaion', 'task', 'ragging', 'hod']
    predicted_label = labels[label_index]

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
