import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from .utils import set_seeds
from .preprocess import create_dataloaders
from .models import SentimentRNN

#Helper functions from train code
def get_optimizer(model, optimizer_name, lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def run_training_for_plots(config, num_epochs=10):
    '''Runs a full training loop for a given config and returns the epoch-by-epoch loss history'''
    print(f"--- Running config: {config['name']} ---")
    
    #1. Setup
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #2. Data
    train_loader, test_loader, vocab_size_actual = create_dataloaders(
        data_path='data/IMDB Dataset.csv',
        vocab_size=10000,
        max_seq_len=config['seq_len'],
        batch_size=32
    )
    
    #3. Model
    model_type = 'LSTM' if config['model_type'] == 'BiLSTM' else config['model_type']
    bidirectional = True if config['model_type'] == 'BiLSTM' else False
    
    model = SentimentRNN(
        vocab_size=vocab_size_actual,
        embedding_dim=100,
        hidden_size=64,
        num_layers=2,
        dropout_prob=0.4,
        model_type=model_type,
        activation_fn='tanh',
        bidirectional=bidirectional).to(device)
    
    #4. Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, config['optimizer'], lr=0.001)
    
    #5. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
              
    print(f"Finished running: {config['name']}")
    return history

def main():
    #Define Best and Worst Configs based on results
    best_config = {
        'name': 'Best (BiLSTM, Adam, 100)',
        'model_type': 'BiLSTM',
        'optimizer': 'Adam',
        'seq_len': 100}
    
    worst_config = {
        'name': 'Worst (LSTM, SGD, 25)',
        'model_type': 'LSTM',
        'optimizer': 'SGD',
        'seq_len': 25}
    
    #Training
    #10 epochs to get smoother curves
    best_history = run_training_for_plots(best_config, num_epochs=10)
    worst_history = run_training_for_plots(worst_config, num_epochs=10)
    
    #Plotting
    sns.set_theme(style="whitegrid")
    epochs = range(1, 11)
    
    #Plot 1: Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, best_history['train_loss'], 'b-o', label=best_config['name'])
    plt.plot(epochs, worst_history['train_loss'], 'r-s', label=worst_config['name'])
    plt.title('Training Loss vs. Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend()
    plt.savefig('results/plots/loss_vs_epochs_TRAIN.png')
    plt.show()
    
    #Plot 2: Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, best_history['val_loss'], 'b-o', label=best_config['name'])
    plt.plot(epochs, worst_history['val_loss'], 'r-s', label=worst_config['name'])
    plt.title('Validation Loss vs. Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.legend()
    plt.savefig('results/plots/loss_vs_epochs_VALIDATION.png')
    plt.show()
    print("Loss plots saved to results/plots/")

if __name__ == '__main__':
    main()
