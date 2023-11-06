import pandas as pd
import json

with open('intents.json', 'r') as json_file:
    json_data = json.load(json_file)

response_records = []

for intent in json_data['intents']:
    tag = intent['tag']
    responses = intent['responses']
    for response in responses:
        response_records.append({'tag': tag, 'response': response})

response_df = pd.DataFrame(response_records)

response_df.to_csv('responses.csv', index=False)
