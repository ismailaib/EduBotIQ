# EduBotIQ

![tenserflow](https://github.com/ismailaib/EduBotIQ/assets/65982831/d8da6f5c-bc63-4fa0-819b-2afb42bcd874)


# EduBotIQ Chat

EduBotIQ Chat is an educational chat application powered by a machine learning model to classify user input and provide relevant responses. This README provides an overview of the project, installation instructions, and other relevant information.

## Overview

The Education ChatBot is an intelligent chatbot designed to assist students in obtaining information about their courses, schedules, grades, and more. This project utilizes TensorFlow for multiclass text classification, natural language understanding, and context management.

### Key Components

- **Flask Web Application (`app.py`):** Handles user interactions through a simple web interface. Users input text, which is then classified by the machine learning model, and the corresponding response is displayed.

- **TensorFlow/Keras Model (`education_classifier_model`):** A deep learning model trained to classify input text into specific educational categories. The model is loaded and used for predictions in the Flask application.

- **Training Data (`training_data.csv`):** Contains labeled educational text data used to train the machine learning model. Each entry has a "text" and "label" field.

- **Responses Data (`responses.csv`):** Contains responses mapped to different educational categories. These responses are displayed to users based on the model's predictions.

## Installation

To run EduBotIQ Chat locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ismailaib/EduBotIQ.git
   cd EduBotIQ
   ```

2. **Install Dependencies:**
   ```bash
   pip install tensorflow
   pip install pandas
   ```

3. **Run the Application:**
   ```bash
   flask run
   ```

   Visit `http://127.0.0.1:5000/` in your web browser to interact with EduBotIQ Chat.

## Model Training (Optional)

If you want to retrain the model with your own data:

1. Replace `training_data.csv` with your dataset.
2. Run the model training script:
   ```bash
   python train_model.py
   ```

   This will save the updated model in the `education_classifier_model` directory.

## Contributors

- [Ismail Aitbouhmd](https://github.com/ismailaib)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
