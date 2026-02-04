"""
Hybrid Model combining Two Towers and Sequence Models.
Uses both user features (age, gender, classes) and activity sequences.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.two_towers import TwoTowerModel, TwoTowerTrainer
from models.sequence_model import ActivityLSTM, SequenceModelTrainer

class HybridModel(nn.Module):
    """
    Hybrid model combining two-tower and sequence approaches.
    """
    
    def __init__(self, two_tower_model, sequence_model, fusion_dim=64):
        super(HybridModel, self).__init__()
        self.two_tower = two_tower_model
        self.sequence = sequence_model
        self.fusion_dim = fusion_dim
        
        # Fusion layer to combine predictions
        self.fusion = nn.Linear(2, 1)  # Combine two scores into one
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_features, activity_ids, activity_sequence=None, 
                activity_classes=None, user_features_seq=None):
        """
        Forward pass combining two-tower and sequence predictions.
        
        Args:
            user_features: User features for two-tower (batch, user_feat_dim)
            activity_ids: Activity IDs for two-tower (batch,)
            activity_sequence: Activity sequence for sequence model (batch, seq_len)
            activity_classes: Activity classes (batch,)
            user_features_seq: User features for sequence model (batch, user_feat_dim)
            
        Returns:
            Combined similarity scores
        """
        # Two-tower prediction
        two_tower_score = self.two_tower(user_features, activity_ids, activity_classes)
        
        # Sequence prediction
        if activity_sequence is not None:
            seq_output = self.sequence(activity_sequence, user_features_seq)
            # Get last prediction and convert to score
            seq_score = seq_output[:, -1, :].max(dim=1)[0]  # Max probability
            # Normalize to similar scale as two-tower
            seq_score = torch.softmax(seq_output[:, -1, :], dim=1).max(dim=1)[0]
        else:
            seq_score = torch.zeros_like(two_tower_score)
        
        # Combine scores
        combined = torch.stack([two_tower_score, seq_score], dim=1)
        fused_score = self.fusion(combined).squeeze()
        
        return fused_score

class HybridTrainer:
    """Trainer for hybrid model."""
    
    def __init__(self, hybrid_model, two_tower_trainer, sequence_trainer, device='cpu'):
        self.model = hybrid_model.to(device)
        self.two_tower_trainer = two_tower_trainer
        self.sequence_trainer = sequence_trainer
        self.device = device
        
    def train(self, user_features_tt, activity_indices, labels, 
              activity_sequences, user_features_seq=None,
              activity_classes=None, epochs=50, batch_size=64, lr=0.001, val_split=0.2):
        """
        Train the hybrid model.
        
        Args:
            user_features_tt: User features for two-tower model
            activity_indices: Activity indices
            labels: Target labels
            activity_sequences: Activity sequences for sequence model
            user_features_seq: User features for sequence model
            activity_classes: Activity class indices
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Validation split ratio
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        if activity_classes is not None:
            (X_user_tt_train, X_user_tt_val, X_act_train, X_act_val, 
             y_train, y_val, X_class_train, X_class_val) = train_test_split(
                user_features_tt, activity_indices, labels, activity_classes,
                test_size=val_split, random_state=42
            )
        else:
            X_user_tt_train, X_user_tt_val, X_act_train, X_act_val, y_train, y_val = train_test_split(
                user_features_tt, activity_indices, labels,
                test_size=val_split, random_state=42
            )
            X_class_train, X_class_val = None, None
        
        # Prepare sequence data
        # For each user-activity pair, get the user's recent activity sequence
        # This is a simplified approach - in practice, you'd want more sophisticated sequence extraction
        seq_train = []
        seq_val = []
        
        # Convert to tensors
        X_user_tt_train = torch.FloatTensor(X_user_tt_train).to(self.device)
        X_user_tt_val = torch.FloatTensor(X_user_tt_val).to(self.device)
        X_act_train = torch.LongTensor(X_act_train).to(self.device)
        X_act_val = torch.LongTensor(X_act_val).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        if activity_classes is not None:
            X_class_train = torch.LongTensor(X_class_train).to(self.device)
            X_class_val = torch.LongTensor(X_class_val).to(self.device)
        
        # For sequence data, we'll use a simple approach: pad sequences
        # In practice, you'd extract actual user sequences
        seq_length = 5
        vocab_size = len(self.sequence_trainer.label_encoder.classes_)
        
        # Create dummy sequences (in practice, extract from activity_sequences)
        seq_train_tensor = torch.zeros(len(X_user_tt_train), seq_length, dtype=torch.long).to(self.device)
        seq_val_tensor = torch.zeros(len(X_user_tt_val), seq_length, dtype=torch.long).to(self.device)
        
        if user_features_seq is not None:
            user_feat_seq_train, user_feat_seq_val = train_test_split(
                user_features_seq, test_size=val_split, random_state=42
            )
            user_feat_seq_train = torch.FloatTensor(user_feat_seq_train).to(self.device)
            user_feat_seq_val = torch.FloatTensor(user_feat_seq_val).to(self.device)
        else:
            user_feat_seq_train, user_feat_seq_val = None, None
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            indices = torch.randperm(len(X_user_tt_train))
            epoch_loss = 0
            
            for i in range(0, len(X_user_tt_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_user_tt = X_user_tt_train[batch_indices]
                batch_act = X_act_train[batch_indices]
                batch_labels = y_train[batch_indices]
                batch_seq = seq_train_tensor[batch_indices]
                
                if activity_classes is not None:
                    batch_class = X_class_train[batch_indices]
                else:
                    batch_class = None
                
                if user_feat_seq_train is not None:
                    batch_user_seq = user_feat_seq_train[batch_indices]
                else:
                    batch_user_seq = None
                
                predictions = self.model(
                    batch_user_tt, batch_act, batch_seq, 
                    batch_class, batch_user_seq
                )
                
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / (len(X_user_tt_train) // batch_size + 1)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                batch_seq_val = seq_val_tensor
                if activity_classes is not None:
                    val_predictions = self.model(
                        X_user_tt_val, X_act_val, batch_seq_val, 
                        X_class_val, user_feat_seq_val
                    )
                else:
                    val_predictions = self.model(
                        X_user_tt_val, X_act_val, batch_seq_val,
                        None, user_feat_seq_val
                    )
                val_loss = criterion(val_predictions, y_val).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses

