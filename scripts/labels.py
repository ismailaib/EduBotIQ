import json

# Load the JSON data
with open('intents.json', 'r') as json_file:
    json_data = json.load(json_file)

# Extract the list of tags
tags = [intent['tag'] for intent in json_data['intents']]

# Print the list of tags
print(tags)
