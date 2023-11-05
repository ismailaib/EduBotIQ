import pandas as pd
import json

# Load the JSON data
with open('intents.json', 'r') as json_file:
    json_data = json.load(json_file)

# Create a list to store the response data
response_records = []

# Extract response data from JSON
for intent in json_data['intents']:
    tag = intent['tag']
    responses = intent['responses']
    for response in responses:
        response_records.append({'tag': tag, 'response': response})

# Create a DataFrame from the response records
response_df = pd.DataFrame(response_records)

# Save the response data to a CSV file
response_df.to_csv('responses.csv', index=False)
