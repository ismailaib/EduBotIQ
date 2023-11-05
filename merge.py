import pandas as pd
import json

# Load the JSON data
with open('intents.json', 'r') as json_file:
    json_data = json.load(json_file)

# Convert JSON data into a format that can be merged with the CSV data
json_records = []
for intent in json_data['intents']:
    for pattern in intent['patterns']:
        json_records.append({'text': pattern, 'label': intent['tag']})

# Create a DataFrame from the JSON records
json_df = pd.DataFrame(json_records)

# Load the CSV data
csv_data = pd.read_csv('training_data.csv')

# Merge the JSON data with the CSV data based on the 'label' column
merged_data = pd.concat([csv_data, json_df], ignore_index=True)

# Save the merged data to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)
