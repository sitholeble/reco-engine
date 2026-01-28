"""
RNN/LSTM Sequence Model for Next Activity Prediction
Predicts the next activity in a sequence based on previous activities.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class ActivityLSTM(nn.Module):
    """
    LSTM model for predicting next activity in a sequence.
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.2):
        super(ActivityLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, sequence_length) with activity indices
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class SequenceModelTrainer:
    """Trainer for sequence prediction model."""
    
    def __init__(self, model, device='cpu'):
        self.device = device
        self.label_encoder = LabelEncoder()
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = None
        
    def prepare_sequences(self, activity_sequences, sequence_length=5):
        """
        Prepare sequences for training.
        
        Args:
            activity_sequences: DataFrame with columns ['user_id', 'activity', 'date']
            sequence_length: Length of input sequences
            
        Returns:
            X: Input sequences (batch_size, sequence_length)
            y: Target sequences (batch_size, sequence_length)
            activity_to_idx: Mapping from activity name to index
        """
        # Sort by user and date
        df = activity_sequences.sort_values(['user_id', 'date']).copy()
        
        # Encode activities
        all_activities = df['activity'].unique()
        self.label_encoder.fit(all_activities)
        df['activity_idx'] = self.label_encoder.transform(df['activity'])
        
        # Create sequences per user
        sequences = []
        for user_id in df['user_id'].unique():
            user_activities = df[df['user_id'] == user_id]['activity_idx'].values
            
            # Create sliding windows
            for i in range(len(user_activities) - sequence_length):
                seq = user_activities[i:i+sequence_length+1]
                sequences.append(seq)
        
        if len(sequences) == 0:
            raise ValueError("Not enough data to create sequences")
        
        sequences = np.array(sequences)
        X = sequences[:, :-1]  # Input sequence
        y = sequences[:, 1:]    # Target sequence (shifted by 1)
        
        # Create activity mapping
        activity_to_idx = {act: idx for idx, act in enumerate(self.label_encoder.classes_)}
        
        return X, y, activity_to_idx
    
    def train(self, X, y, epochs=50, batch_size=64, lr=0.001, val_split=0.2):
        """Train the model."""
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.LongTensor(X_train).to(self.device)
        X_val = torch.LongTensor(X_val).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Batch training
            indices = torch.randperm(len(X_train))
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Forward pass
                predictions = self.model(batch_X)  # (batch, seq_len, vocab_size)
                
                # Reshape for loss: (batch * seq_len, vocab_size) and (batch * seq_len,)
                predictions = predictions.reshape(-1, predictions.size(-1))
                targets = batch_y.reshape(-1)
                
                loss = criterion(predictions, targets)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / (len(X_train) // batch_size + 1)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_predictions = val_predictions.reshape(-1, val_predictions.size(-1))
                val_targets = y_val.reshape(-1)
                val_loss = criterion(val_predictions, val_targets).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict_next(self, activity_sequence, top_k=5):
        """
        Predict next activity given a sequence of previous activities.
        
        Args:
            activity_sequence: List of activity names (e.g., ['cycling', 'spin_class', 'weight_training'])
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (activity, probability)
        """
        self.model.eval()
        
        # Encode sequence
        try:
            encoded_seq = self.label_encoder.transform(activity_sequence)
        except ValueError as e:
            # Handle unknown activities
            print(f"Warning: Some activities not seen during training: {e}")
            return []
        
        # Convert to tensor
        seq_tensor = torch.LongTensor([encoded_seq]).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(seq_tensor)  # (1, seq_len, vocab_size)
            # Get prediction for last position
            last_prediction = predictions[0, -1, :]  # (vocab_size,)
            probabilities = torch.softmax(last_prediction, dim=0)
        
        # Get top k
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
        
        results = [
            (self.label_encoder.inverse_transform([idx.item()])[0], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return results
    
    def predict_sequence(self, activity_sequence, length=5):
        """
        Predict a sequence of next activities.
        
        Args:
            activity_sequence: Initial sequence of activity names
            length: Length of sequence to predict
            
        Returns:
            List of predicted activity names
        """
        current_sequence = activity_sequence.copy()
        predictions = []
        
        for _ in range(length):
            next_activity = self.predict_next(current_sequence, top_k=1)
            if next_activity:
                predicted_activity = next_activity[0][0]
                predictions.append(predicted_activity)
                current_sequence.append(predicted_activity)
                # Keep sequence length manageable
                if len(current_sequence) > 10:
                    current_sequence = current_sequence[-10:]
            else:
                break
        
        return predictions

