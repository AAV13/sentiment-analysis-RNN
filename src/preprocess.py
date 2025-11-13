import pandas as pd
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class IMDbDataProcessor:
    def __init__(self, data_path, vocab_size=10000):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.tokenizer = word_tokenize
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False

    def clean_text(self, text):
        '''Lowercase, remove punctuation, and remove HTML tags'''
        text = text.lower()
        #Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        #Remove punctuation and special characters (keep only words and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        #Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_vocab(self, train_texts):
        '''Builds the vocabulary from the training texts.'''
        print("Building vocabulary...")
        all_tokens = []
        for text in tqdm(train_texts):
            all_tokens.extend(self.tokenizer(text))
        
        word_counts = Counter(all_tokens)
        most_common_words = word_counts.most_common(self.vocab_size - 2) #-2 for <PAD> and <UNK>
        
        #Define special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common_words):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word
            
        self.vocab_built = True
        print(f"Vocabulary built. Size: {len(self.word_to_idx)}")

    def text_to_sequence(self, text):
        '''Converts a single text string to a sequence of token IDs.'''
        if not self.vocab_built:
            raise Exception("Vocabulary not built. Call build_vocab() first.")
            
        tokens = self.tokenizer(text)
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        return sequence

    def load_and_preprocess_data(self):
        """
        Main function to load, split, and process the dataset.
        """
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path)
        
        #Map labels
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        #Clean text
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        #Split data (using the 50/50 split described in the dataset)
        #Assuming the first 25k are train and the next 25k are test
        train_df = df.iloc[:25000]
        test_df = df.iloc[25000:]
        
        #Build vocabulary *only* on the training data
        self.build_vocab(train_df['cleaned_review'])
        
        #Convert all texts to sequences
        X_train_seq = [self.text_to_sequence(text) for text in train_df['cleaned_review']]
        X_test_seq = [self.text_to_sequence(text) for text in test_df['cleaned_review']]
        
        y_train = train_df['sentiment'].values
        y_test = test_df['sentiment'].values
        
        return X_train_seq, y_train, X_test_seq, y_test

class IMDbDataset(Dataset):
    def __init__(self, sequences, labels, max_seq_len):
        self.sequences = sequences
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        #Pad or truncate
        if len(seq) > self.max_seq_len:
            padded_seq = seq[:self.max_seq_len]
        else:
            padded_seq = seq + [0] * (self.max_seq_len - len(seq)) #0 is <PAD>
            
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def create_dataloaders(data_path, vocab_size, max_seq_len, batch_size):
    """
    High-level function to get train and test dataloaders.
    """
    processor = IMDbDataProcessor(data_path, vocab_size)
    X_train_seq, y_train, X_test_seq, y_test = processor.load_and_preprocess_data()
    
    train_dataset = IMDbDataset(X_train_seq, y_train, max_seq_len)
    test_dataset = IMDbDataset(X_test_seq, y_test, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Return vocab_size for model embedding layer
    vocab_size = len(processor.word_to_idx)
    
    return train_loader, test_loader, vocab_size

if __name__ == '__main__':
    #Test the preprocessor
    DATA_PATH = '../data/IMDB Dataset.csv'
    VOCAB_SIZE = 10000
    MAX_LEN = 50
    BATCH_SIZE = 32
    
    train_loader, test_loader, vocab_size = create_dataloaders(
        data_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_LEN,
        batch_size=BATCH_SIZE)
    
    print(f"\nCreated loaders. Vocab size: {vocab_size}")
    
    #Check one batch
    for batch in train_loader:
        inputs, labels = batch
        print(f"Batch inputs shape: {inputs.shape}") #[batch_size, max_len]
        print(f"Batch labels shape: {labels.shape}") #[batch_size]
        print(f"First sequence: {inputs[0]}")
        print(f"First label: {labels[0]}")
        break
