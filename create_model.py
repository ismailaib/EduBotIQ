import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load training data from a CSV file
data = pd.read_csv('training_data.csv')

# Extract text and labels from the CSV
texts = data['text'].tolist()
labels = data['label'].tolist()

# Tokenize the text data and specify labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to have the same length
max_sequence_length = 10  # Adjust as needed
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# One-hot encode labels and specify labels
labels_set = set(labels)
num_classes = len(labels_set)
label_to_index = {label: i for i, label in enumerate(labels_set)}
index_to_label = {i: label for label, i in label_to_index.items()}
labels = [label_to_index[label] for label in labels]

labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Build the model
model = keras.Sequential([
    keras.layers.Embedding(len(word_index) + 1, 128, input_length=max_sequence_length),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=20)

# Save the model
# Save the model in the recommended Keras format
model.save('education_classifier_model_v2')  # No file extension needed, it will be saved as a directory

# Save labels mapping to a JSON file for use in app.py
import json
with open('labels_mapping.json', 'w') as json_file:
    json.dump(label_to_index, json_file)
