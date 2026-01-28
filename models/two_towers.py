"""
Two Towers Model for Recommendations
One tower for user features, one tower for item (activity) features.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class UserTower(nn.Module):
    """Tower for encoding user features."""
    
    def __init__(self, user_feature_dim, embedding_dim=64):
        super(UserTower, self).__init__()
        self.fc1 = nn.Linear(user_feature_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ActivityTower(nn.Module):
    """Tower for encoding activity features."""
    
    def __init__(self, num_activities, embedding_dim=64):
        super(ActivityTower, self).__init__()
        # Embedding layer for activities
        self.embedding = nn.Embedding(num_activities, embedding_dim)
        
    def forward(self, activity_ids):
        return self.embedding(activity_ids)

class TwoTowerModel(nn.Module):
    """
    Two Towers Model for recommendations.
    Learns embeddings for users and activities separately, then computes similarity.
    """
    
    def __init__(self, user_feature_dim, num_activities, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(user_feature_dim, embedding_dim)
        self.activity_tower = ActivityTower(num_activities, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, user_features, activity_ids):
        """
        Forward pass.
        
        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim)
            activity_ids: Tensor of shape (batch_size,) with activity indices
            
        Returns:
            Similarity scores (dot product of embeddings)
        """
        user_emb = self.user_tower(user_features)
        activity_emb = self.activity_tower(activity_ids)
        
        # Compute dot product similarity
        similarity = (user_emb * activity_emb).sum(dim=1)
        return similarity
    
    def get_user_embedding(self, user_features):
        """Get user embedding for a given user."""
        return self.user_tower(user_features)
    
    def get_activity_embedding(self, activity_ids):
        """Get activity embeddings."""
        return self.activity_tower(activity_ids)

class TwoTowerTrainer:
    """Trainer for Two Towers model."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
    def prepare_data(self, user_profiles, interaction_matrix, activity_to_idx):
        """
        Prepare data for training.
        
        Args:
            user_profiles: DataFrame with user features
            interaction_matrix: DataFrame with user_id, activity, count/rating
            activity_to_idx: Mapping from activity name to index
        """
        # Merge user profiles with interactions
        data = interaction_matrix.merge(
            user_profiles[['user_id', 'age', 'weight_kg', 'height_cm']],
            on='user_id',
            how='left'
        )
        
        # Create activity indices
        data['activity_idx'] = data['activity'].map(activity_to_idx)
        
        # Prepare features
        user_features = data[['age', 'weight_kg', 'height_cm']].values
        user_features = self.scaler.fit_transform(user_features)
        
        activity_indices = data['activity_idx'].values
        labels = data['count' if 'count' in data.columns else 'avg_rating'].values
        
        # Normalize labels to [0, 1] for binary cross-entropy or use as-is for MSE
        if labels.max() > 1:
            labels = labels / labels.max()
        
        return user_features, activity_indices, labels
    
    def train(self, user_features, activity_indices, labels, 
              epochs=50, batch_size=64, lr=0.001, val_split=0.2):
        """Train the model."""
        # Split data
        X_user_train, X_user_val, X_act_train, X_act_val, y_train, y_val = train_test_split(
            user_features, activity_indices, labels,
            test_size=val_split, random_state=42
        )
        
        # Convert to tensors
        X_user_train = torch.FloatTensor(X_user_train).to(self.device)
        X_user_val = torch.FloatTensor(X_user_val).to(self.device)
        X_act_train = torch.LongTensor(X_act_train).to(self.device)
        X_act_val = torch.LongTensor(X_act_val).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Batch training
            indices = torch.randperm(len(X_user_train))
            epoch_loss = 0
            
            for i in range(0, len(X_user_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_user = X_user_train[batch_indices]
                batch_act = X_act_train[batch_indices]
                batch_labels = y_train[batch_indices]
                
                predictions = self.model(batch_user, batch_act)
                loss = criterion(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / (len(X_user_train) // batch_size + 1)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_user_val, X_act_val)
                val_loss = criterion(val_predictions, y_val).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses
    
    def recommend(self, user_features, activity_to_idx, top_k=5, exclude_activities=None):
        """
        Recommend activities for a user.
        
        Args:
            user_features: Array of shape (user_feature_dim,) - [age, weight, height]
            activity_to_idx: Mapping from activity name to index
            top_k: Number of recommendations
            exclude_activities: Set of activity names to exclude
            
        Returns:
            List of tuples (activity, score)
        """
        self.model.eval()
        
        # Normalize user features
        user_features = self.scaler.transform([user_features])
        user_tensor = torch.FloatTensor(user_features).to(self.device)
        
        # Get user embedding
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_tensor)
        
        # Score all activities
        idx_to_activity = {v: k for k, v in activity_to_idx.items()}
        scores = {}
        
        for activity, act_idx in activity_to_idx.items():
            if exclude_activities and activity in exclude_activities:
                continue
            
            act_tensor = torch.LongTensor([act_idx]).to(self.device)
            with torch.no_grad():
                act_emb = self.model.get_activity_embedding(act_tensor)
                score = (user_emb * act_emb).sum().item()
            
            scores[activity] = score
        
        # Get top k
        top_activities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_activities

