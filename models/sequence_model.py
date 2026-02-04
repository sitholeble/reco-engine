"""
RNN/LSTM Sequence Model for Next Activity Prediction
Predicts the next activity in a sequence based on previous + current activities.
Enhanced to include user features (age, gender, activity classes).
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
    Enhanced to use previous + current activities to predict future.
    Can also incorporate user features (age, gender, classes).
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, 
                 dropout=0.2, user_feature_dim=0):
        super(ActivityLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.user_feature_dim = user_feature_dim
        
        # If user features are provided, concatenate them with embeddings
        lstm_input_dim = embedding_dim
        if user_feature_dim > 0:
            self.user_feature_proj = nn.Linear(user_feature_dim, embedding_dim)
            lstm_input_dim = embedding_dim + embedding_dim  # activity emb + user feature emb
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, user_features=None):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, sequence_length) with activity indices
            user_features: Optional tensor of shape (batch_size, user_feature_dim) with user features
            
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # If user features provided, concatenate them to each time step
        if user_features is not None and self.user_feature_dim > 0:
            user_feat_emb = self.user_feature_proj(user_features)  # (batch, embedding_dim)
            # Expand to match sequence length: (batch, 1, embedding_dim) -> (batch, seq_len, embedding_dim)
            user_feat_emb = user_feat_emb.unsqueeze(1).expand(-1, embedded.size(1), -1)
            embedded = torch.cat([embedded, user_feat_emb], dim=2)  # (batch, seq_len, embedding_dim*2)
        
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
        
    def prepare_sequences(self, activity_sequences, user_profiles=None, sequence_length=5, 
                          use_user_features=False):
        """
        Prepare sequences for training.
        Uses previous + current activities to predict future.
        
        Args:
            activity_sequences: DataFrame with columns ['user_id', 'activity', 'date']
            user_profiles: Optional DataFrame with user features (age, gender, etc.)
            sequence_length: Length of input sequences (previous + current)
            use_user_features: Whether to include user features (age, gender, classes)
            
        Returns:
            X: Input sequences (batch_size, sequence_length)
            y: Target sequences (batch_size, sequence_length) - future activities
            user_features: Optional user features (batch_size, user_feature_dim)
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
        user_feature_list = []
        
        for user_id in df['user_id'].unique():
            user_activities = df[df['user_id'] == user_id]['activity_idx'].values
            
            # Create sliding windows: use previous + current to predict future
            # For sequence_length=5: use activities [0:5] to predict [1:6]
            # This means we use previous 4 + current 1 to predict next 5
            for i in range(len(user_activities) - sequence_length):
                # Input: previous + current (sequence_length activities)
                # Output: future activities (shifted by 1, so we predict what comes after current)
                seq = user_activities[i:i+sequence_length+1]
                sequences.append(seq)
                
                # Extract user features if needed
                if use_user_features and user_profiles is not None:
                    user_row = user_profiles[user_profiles['user_id'] == user_id]
                    if len(user_row) > 0:
                        user_row = user_row.iloc[0]
                        # Extract age, gender, and activity class preferences
                        features = []
                        if 'age' in user_row:
                            features.append(user_row['age'])
                        if 'gender' in user_row:
                            # Encode gender: M=1, F=0, Other=0.5
                            gender_val = 1.0 if user_row['gender'] == 'M' else (0.5 if user_row['gender'] == 'Other' else 0.0)
                            features.append(gender_val)
                        # Add activity class preferences if available
                        if 'favorite_activities' in user_row and pd.notna(user_row['favorite_activities']):
                            fav_activities = str(user_row['favorite_activities']).split(',')
                            # Count classes in favorites
                            from data_generator import ACTIVITY_TO_CLASS
                            class_counts = {'cardio': 0, 'strength': 0, 'flexibility': 0, 'sports': 0}
                            for act in fav_activities:
                                act_class = ACTIVITY_TO_CLASS.get(act.strip(), 'other')
                                if act_class in class_counts:
                                    class_counts[act_class] += 1
                            # Normalize
                            total = sum(class_counts.values()) if sum(class_counts.values()) > 0 else 1
                            features.extend([class_counts['cardio']/total, class_counts['strength']/total,
                                            class_counts['flexibility']/total, class_counts['sports']/total])
                        else:
                            features.extend([0.25, 0.25, 0.25, 0.25])  # Default uniform distribution
                        
                        user_feature_list.append(features)
                    else:
                        if use_user_features:
                            user_feature_list.append([35, 0.5, 0.25, 0.25, 0.25, 0.25])  # Default
                elif use_user_features:
                    user_feature_list.append([35, 0.5, 0.25, 0.25, 0.25, 0.25])  # Default
        
        if len(sequences) == 0:
            raise ValueError("Not enough data to create sequences")
        
        sequences = np.array(sequences)
        X = sequences[:, :-1]  # Input: previous + current (sequence_length activities)
        y = sequences[:, 1:]    # Target: future activities (shifted by 1)
        
        # Create activity mapping
        activity_to_idx = {act: idx for idx, act in enumerate(self.label_encoder.classes_)}
        
        if use_user_features and user_profiles is not None and len(user_feature_list) > 0:
            user_features = np.array(user_feature_list)
            return X, y, user_features, activity_to_idx
        else:
            # Return None for user_features if not used
            return X, y, None, activity_to_idx
    
    def train(self, X, y, epochs=50, batch_size=64, lr=0.001, val_split=0.2, user_features=None):
        """Train the model."""
        # Split data
        from sklearn.model_selection import train_test_split
        
        if user_features is not None:
            X_train, X_val, y_train, y_val, user_feat_train, user_feat_val = train_test_split(
                X, y, user_features, test_size=val_split, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42
            )
            user_feat_train, user_feat_val = None, None
        
        # Convert to tensors
        X_train = torch.LongTensor(X_train).to(self.device)
        X_val = torch.LongTensor(X_val).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        if user_features is not None:
            user_feat_train = torch.FloatTensor(user_feat_train).to(self.device)
            user_feat_val = torch.FloatTensor(user_feat_val).to(self.device)
        
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
                if user_features is not None:
                    batch_user_feat = user_feat_train[batch_indices]
                    predictions = self.model(batch_X, batch_user_feat)  # (batch, seq_len, vocab_size)
                else:
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
                if user_features is not None:
                    val_predictions = self.model(X_val, user_feat_val)
                else:
                    val_predictions = self.model(X_val)
                val_predictions = val_predictions.reshape(-1, val_predictions.size(-1))
                val_targets = y_val.reshape(-1)
                val_loss = criterion(val_predictions, val_targets).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict_next(self, activity_sequence, top_k=5, user_features=None):
        """
        Predict next activity given a sequence of previous + current activities.
        
        Args:
            activity_sequence: List of activity names (e.g., ['cycling', 'spin_class', 'weight_training'])
            top_k: Number of top predictions to return
            user_features: Optional user features array (age, gender, class preferences)
            
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
            if user_features is not None:
                user_feat_tensor = torch.FloatTensor([user_features]).to(self.device)
                predictions = self.model(seq_tensor, user_feat_tensor)  # (1, seq_len, vocab_size)
            else:
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

