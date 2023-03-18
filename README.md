# BERT-based_Chatbot
This is a chatbot built using the BERT (Bidirectional Encoder Representations from Transformers) model. The chatbot is trained to respond to user queries based on predefined categories such as greetings, weather-related questions, and jokes.

# Requirements
- Python 3.6 or later
- PyTorch 1.9.0 or later
- Transformers 4.8.1 or later
# How to Use

Code1: chatbot.py
- The chatbot.py file contains the Chatbot class that defines the architecture of the chatbot. The class inherits from nn.Module and contains a BERT model, an LSTM layer, and a linear layer. The generate_response method generates a response to an input query using a pre-trained BERT language model.

Code2: train_chatbot.py
- The train_chatbot.py file trains the chatbot using a custom-built classifier on top of the pre-trained BERT model. The script defines a CustomTextDataset class that loads the training data and a Chatbot class that inherits from nn.Module and contains the custom-built classifier.

Code3: data_loader.py
- The data_loader.py file contains the CustomTextDataset class used to load the data for training the chatbot. The data consists of a list of tuples with input text and target labels. The Dataset class provided by PyTorch is used to create a custom dataset, which is then split into training and validation sets.

# Data
The training data should be in a CSV file with the following columns:
```
input_text,target
How are you?,0
Hello, how's it going?,0
Hey, what's up?,0
What's the weather like?,1
Is it going to rain today?,1
Do I need an umbrella?,1
Tell me a joke,2
Why did the chicken cross the road?,2
Can you make me laugh?,2
... 
```
Here, input_text refers to the user's query, and target is the integer label representing the category. In this example, 0 represents greetings, 1 represents weather-related questions, and 2 represents jokes. You can define your own set of categories and label them accordingly.

# Model
The chatbot model is defined in chatbot.py and consists of three parts:

- BERT model for encoding input text
- LSTM layers for learning the sequence
- Linear layer for combining the encoded input text and the output of LSTM layers
The model is pre-trained on the BERT base uncased model and fine-tuned for the chatbot task.

# Training
The model is trained using the train.py script, which loads the data from the CSV file, preprocesses it, and trains the model on it. The script fine-tunes the BERT model on the chatbot task by freezing the pre-trained layers and training only the custom-built layers. After training the custom-built layers, the pre-trained layers are unfrozen, and the entire model is trained on the chatbot task.

# Inference
After training, the model can be used to generate responses to user queries using the generate_response() method defined in chatbot.py. This method takes a user query as input and returns the model's predicted response. The generate_response() method uses the BERT model's mask language modeling (MLM) head for text generation.

# Example
Here's an example usage of the chatbot:
```
import torch
from chatbot import Chatbot

# Load the trained model
chatbot = Chatbot()
model_dir = '/path/to/trained/models/'
chatbot.load_state_dict(torch.load(model_dir + 'chatbot_state_dict_transfer.pth'))
chatbot.eval()

# Generate a response to user query
input_query = "What's the weather like today?"
response = chatbot.generate_response(input_query)

# Print the response
print(response)
```
# Acknowledgements
This project was inspired by the <a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html" target="_blank">Pytorch tutorial</a> on fine-tuning the BERT model for text classification.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
