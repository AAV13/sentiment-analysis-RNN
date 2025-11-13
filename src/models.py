import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    '''A flexible RNN model for sentiment classification. Can be configured as a simple RNN, LSTM, or Bidirectional LSTM.'''
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_size, 
                 num_layers, 
                 dropout_prob,
                 model_type='LSTM', 
                 activation_fn='tanh',
                 bidirectional=False):
        
        super(SentimentRNN, self).__init__()
        
        self.model_type = model_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        #Layers
        #1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #2. Recurrent Layer (RNN, LSTM)
        #Get the appropriate activation function for simple RNN
        if activation_fn.lower() == 'relu':
            self.activation = 'relu'
        elif activation_fn.lower() == 'sigmoid':
            #Sigmoid is a very poor choice for RNNs (vanishing gradients)
            self.activation = 'tanh' #Defaulting to tanh if sigmoid is passed
            print("Warning: Sigmoid is not a standard 'nonlinearity' for nn.RNN. Using Tanh.")
        else:
            self.activation = 'tanh'
            
        
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=self.activation, #'tanh' or 'relu'
                dropout=dropout_prob if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=self.bidirectional)
            
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=self.bidirectional)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'RNN' or 'LSTM'.")

        #3. Dropout Layer (for after RNN)
        self.dropout = nn.Dropout(dropout_prob)
        
        #4. Fully Connected Output Layer
        #Multiply hidden_size by 2 if bidirectional
        fc_input_size = hidden_size * 2 if self.bidirectional else hidden_size
        
        self.fc = nn.Linear(fc_input_size, 1)
        
        #5. Sigmoid Activation for output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x shape: (batch_size, seq_len)
        
        #1. Embedding
        embedded = self.embedding(x)
        #embedded shape: (batch_size, seq_len, embedding_dim)
        
        #2. RNN/LSTM Layer
        #rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        #hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        
        if self.model_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else: #RNN
            rnn_out, hidden = self.rnn(embedded)
            
        #Handle Bidirectional vs. Unidirectional
        
        if self.bidirectional:
            #We want the last hidden state from *both* directions
            #hidden shape is (num_layers * 2, batch_size, hidden_size)
            #We take the last layer's forward (idx -2) and backward (idx -1) hidden states
            last_hidden_fwd = hidden[-2, :, :]
            last_hidden_bwd = hidden[-1, :, :]
            #Concatenate them
            hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
            #hidden shape: (batch_size, hidden_size * 2)
        else:
            #We just want the last hidden state from the last layer
            #hidden shape is (num_layers, batch_size, hidden_size)
            hidden = hidden[-1, :, :]
            #hidden shape: (batch_size, hidden_size)

        #3. Dropout
        hidden = self.dropout(hidden)
        
        #4. Fully Connected Layer
        out = self.fc(hidden)
        #out shape: (batch_size, 1)
        
        #5. Sigmoid Activation
        out = self.sigmoid(out)
        
        #Squeeze to match label shape
        return out.squeeze() #shape: (batch_size)


if __name__ == '__main__':
    #Test Cases
    
    #Project parameters
    VOCAB_SIZE = 10000
    EMBED_DIM = 100
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    BATCH_SIZE = 32
    SEQ_LEN = 50
    
    #1. Test Simple RNN (Unidirectional)
    print("--- Testing Unidirectional RNN (ReLU) ---")
    rnn_model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        model_type='RNN',
        activation_fn='relu',
        bidirectional=False)
    
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output = rnn_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE,)

    #2. Test LSTM (Unidirectional)
    print("\n--- Testing Unidirectional LSTM ---")
    lstm_model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        model_type='LSTM',
        bidirectional=False)
    output = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE,)

    #3. Test Bidirectional LSTM
    print("\n--- Testing Bidirectional LSTM ---")
    bilstm_model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        model_type='LSTM',
        bidirectional=True)
    
    output = bilstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE,)
    print("\nAll models passed forward pass test.")
