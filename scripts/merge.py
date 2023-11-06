import pandas as pd
import json

with open('intents.json', 'r') as json_file:
    json_data = json.load(json_file)

json_records = []
for intent in json_data['intents']:
    for pattern in intent['patterns']:
        json_records.append({'text': pattern, 'label': intent['tag']})

json_df = pd.DataFrame(json_records)

csv_data = pd.read_csv('training_data.csv')

merged_data = pd.concat([csv_data, json_df], ignore_index=True)

merged_data.to_csv('merged_data.csv', index=False)
