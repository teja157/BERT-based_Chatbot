import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertLMHeadModel, TextGenerationPipeline

class Chatbot(nn.Module):
    def __init__(self):
        super(Chatbot, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=256, num_layers=2, batch_first=True)
        self.linear = nn.Linear(self.bert.config.hidden_size + 256, self.bert.config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.mlm = BertLMHeadModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, hidden):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm_output, hidden = self.lstm(bert_output, hidden)
        lstm_output = self.dropout(lstm_output[:, -1, :])
        combined_output = torch.cat([bert_output.mean(dim=1), lstm_output], dim=-1)
        combined_output = self.dropout(nn.functional.relu(self.linear(combined_output)))
        return combined_output, hidden

    def generate_response(self, input_query):
        self.mlm.to('cpu')  # Move the model to CPU
        text_generation = TextGenerationPipeline(self.mlm, self.tokenizer)
        response = text_generation(input_query, max_length=50, do_sample=True, temperature=0.8)
        return response[0]['generated_text']

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, 256).to('cuda'), torch.zeros(2, batch_size, 256).to('cuda'))

# Example usage:
chatbot = Chatbot().to('cuda')
input_query = "Hello, how are you today?"
response = chatbot.generate_response(input_query)
model_dir = 'model/path'
torch.save(chatbot.state_dict(), model_dir + 'chatbot_state_dict.pth')

print(response)  
