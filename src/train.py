import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import time
import os
from sklearn.metrics import accuracy_score, f1_score
from .utils import set_seeds
from .preprocess import create_dataloaders
from .models import SentimentRNN

def get_optimizer(model, optimizer_name, lr):
    """Factory function for creating an optimizer."""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None):
    """Performs one full training epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        #Zero gradients
        optimizer.zero_grad()
        #Forward pass
        outputs = model(inputs)
        #Calculate loss
        loss = criterion(outputs, labels)
        #Backward pass
        loss.backward()
        #Gradient Clipping (if specified)
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
        #Update weights
        optimizer.step()
        total_loss += loss.item()
        
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    return avg_loss, epoch_time

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            #Convert outputs to predictions (0 or 1)
            preds = (outputs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def main():
    parser = argparse.ArgumentParser(description="RNN Sentiment Classification Experiment")
    
    #Experiment Configuration
    parser.add_argument('--model_type', type=str, default='LSTM', help="RNN, LSTM")
    parser.add_argument('--activation', type=str, default='tanh', help="tanh, relu, sigmoid")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Adam, SGD, RMSProp")
    parser.add_argument('--seq_len', type=int, default=50, help="25, 50, 100")
    parser.add_argument('--grad_clip', type=float, default=None, help="Max norm for gradient clipping (e.g., 1.0)")
    
    #Model Hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help="Top N words to keep")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4) # Mid-range of 0.3-0.5
    parser.add_argument('--bidirectional', action='store_true', help="Use Bidirectional LSTM/RNN")
    
    #Training Settings
    parser.add_argument('--epochs', type=int, default=5) # 5 epochs is good for speed
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    #Setup
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Handle Bidirectional LSTM logic
    #The prompt separates "Bidirectional LSTM" as an "Architecture"
    if args.model_type.lower() == 'bilstm':
        args.model_type = 'LSTM'
        args.bidirectional = True
    
    print(f"--- Starting Experiment ---")
    print(f"Config: Model={args.model_type}, Bi={args.bidirectional}, Act={args.activation}, "
          f"Optim={args.optimizer}, SeqLen={args.seq_len}, Clip={args.grad_clip}")
    print(f"Device: {device}")
    
    #Data
    #Path is relative to the project root (SentimentRNN)
    DATA_PATH = 'data/IMDB Dataset.csv'
    train_loader, test_loader, vocab_size_actual = create_dataloaders(
        data_path=DATA_PATH,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size)
    print(f"Data loaded. Actual vocab size: {vocab_size_actual}")

    #Model
    model = SentimentRNN(
        vocab_size=vocab_size_actual,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_prob=args.dropout,
        model_type=args.model_type,
        activation_fn=args.activation,
        bidirectional=args.bidirectional).to(device)
    
    print(f"Model created:\n{model}")
    
    #Loss and Optimizer
    criterion = nn.BCELoss() # Binary Cross-Entropy
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    
    #Training Loop
    best_val_f1 = 0
    results = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        train_loss, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        
        val_loss, val_acc, val_f1 = evaluate(
            model, test_loader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s")
        
        # Store results
        results.append({
            'Model': 'BiLSTM' if args.bidirectional else args.model_type,
            'Activation': args.activation,
            'Optimizer': args.optimizer,
            'Seq Length': args.seq_len,
            'Grad Clipping': 'Yes' if args.grad_clip else 'No',
            'Accuracy': val_acc,
            'F1': val_f1,
            'Epoch Time (s)': epoch_time})
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
    #Save Results
    final_result = results[-1] # Get the last epoch's result
    print("\n--- Final Epoch Metrics ---")
    print(final_result)
    
    df = pd.DataFrame([final_result])
    
    # Path is relative to project root
    results_file = 'results/metrics.csv'
    if not os.path.exists(results_file):
        df.to_csv(results_file, index=False)
    else:
        df.to_csv(results_file, mode='a', header=False, index=False)
    
    print(f"Results appended to {results_file}")

if __name__ == '__main__':
    main()
