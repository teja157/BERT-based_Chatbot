import torch.optim as optim

chatbot = Chatbot()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chatbot.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(chatbot.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Load the pre-trained models and freeze the pre-trained layers
chatbot.bert = BertModel.from_pretrained('bert-base-uncased')
chatbot.bert.requires_grad_(False)
chatbot.lstm = nn.LSTM(input_size=chatbot.bert.config.hidden_size, hidden_size=256, num_layers=2, batch_first=True)

num_epochs = 50
batch_size = 16

# Prepare your training and validation datasets here
train_loader = url(
     "https://datasets-server.huggingface.co/rows?dataset=rajuptvs%2Fecommerce_products_clip&config=default&split=train&offset=0&length=100")
val_loader = url(
     "https://datasets-server.huggingface.co/rows?dataset=rajuptvs%2Fecommerce_products_clip&config=default&split=train&offset=0&length=100")

# Define preprocess_inputs, preprocess_targets, accuracy, and evaluate_model functions here
def preprocess_inputs(inputs):
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    return input_ids, attention_mask

def preprocess_targets(targets):
    return targets.to(device)

def accuracy(outputs, targets):
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.shape[0]
    return correct / total

def evaluate_model(chatbot, val_loader, loss_fn):
    chatbot.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            input_ids, attention_mask = preprocess_inputs(inputs)
            targets = preprocess_targets(targets)
            outputs, _ = chatbot(input_ids, attention_mask, chatbot.init_hidden(batch_size))
            loss = loss_fn(outputs, targets)
            acc = accuracy(outputs, targets)

            val_loss += loss.item()
            val_acc += acc
            val_batches += 1

    chatbot.train()
    return val_loss / val_batches, val_acc / val_batches

# Train the custom-built layers
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0

    for inputs, targets in train_loader:
        # Preprocess the inputs and targets
        input_ids, attention_mask = preprocess_inputs(inputs)
        targets = preprocess_targets(targets)

        # Forward pass
        outputs, _ = chatbot(input_ids, attention_mask, chatbot.init_hidden(batch_size))

        # Compute the loss and accuracy
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the training statistics
        train_loss += loss.item()
        train_acc += acc
        train_batches += 1

    # Evaluate the model on the validation set
    val_loss, val_acc = evaluate_model(chatbot, val_loader, loss_fn)

    # Print the training statistics
    print(f'Epoch {epoch + 1}: Train Loss={train_loss/train_batches:.4f}, Train Acc={train_acc/train_batches:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

# Fine-tune the pre-trained layers
chatbot.bert.requires_grad_(True)

# Train the entire model
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0

    for inputs, targets in train_loader:
        # Preprocess the inputs and targets
        input_ids, attention_mask = preprocess_inputs(inputs)
        targets = preprocess_targets(targets)

        # Forward pass
        outputs, _ = chatbot(input_ids, attention_mask, chatbot.init_hidden(batch_size))

        # Compute the loss and accuracy
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the training statistics
        train_loss += loss.item()
        train_acc += acc
        train_batches += 1

    # Evaluate the model on the validation set
    val_loss, val_acc = evaluate_model(chatbot, val_loader, loss_fn)

    # Print the training statistics
    print(f'Epoch {epoch + 1}: Train Loss={train_loss/train_batches:.4f}, Train Acc={train_acc/train_batches:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

# Fine-tune the pre-trained layers
chatbot.bert.requires_grad_(True)

# Train the entire model
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_batches = 0

    for inputs, targets in train_loader:
        # Preprocess the inputs and targets
        input_ids, attention_mask = preprocess_inputs(inputs)
        targets = preprocess_targets(targets)

        # Forward pass
        outputs, _ = chatbot(input_ids, attention_mask, chatbot.init_hidden(batch_size))

        # Compute the loss and accuracy
        loss = loss_fn(outputs, targets)
        acc = accuracy(outputs, targets)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the training statistics
        train_loss += loss.item()
        train_acc += acc
        train_batches += 1

    # Evaluate the model on the validation set
    val_loss, val_acc = evaluate_model(chatbot, val_loader, loss_fn)

    # Print the training statistics
    print(f'Epoch {epoch + 1}: Train Loss={train_loss/train_batches:.4f}, Train Acc={train_acc/train_batches:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

# Save the trained model
model_dir = 'model/path'
torch.save(chatbot.state_dict(), model_dir + 'chatbot_state_dict_transfer.pth')
