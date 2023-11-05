import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load training data from a CSV file
data = pd.read_csv('training_data.csv')

texts = data['text'].tolist()
labels = data['label'].tolist() 


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)


max_sequence_length = 10 
data = pad_sequences(sequences, maxlen=max_sequence_length)

# One-hot encode labels
num_classes = len(set(labels))
label_to_index = {label: i for i, label in enumerate(set(labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
labels = [label_to_index[label] for label in labels]
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Build the model
model = keras.Sequential([
    keras.layers.Embedding(len(word_index) + 1, 128, input_length=max_sequence_length),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=20) 

# Save the model in the recommended Keras format
model.save('education_classifier_model')  # No file extension needed, it will be saved as a directory

# Load the model later using the same format
loaded_model = keras.models.load_model('education_classifier_model')
