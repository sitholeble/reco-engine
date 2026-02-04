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
    """Tower for encoding user features (age, gender, activity classes)."""
    
    def __init__(self, user_feature_dim, embedding_dim=64):
        super(UserTower, self).__init__()
        # Enhanced architecture for better feature learning
        self.fc1 = nn.Linear(user_feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ActivityTower(nn.Module):
    """Tower for encoding activity features (can include activity classes)."""
    
    def __init__(self, num_activities, embedding_dim=64, num_classes=0):
        super(ActivityTower, self).__init__()
        # Embedding layer for activities
        self.embedding = nn.Embedding(num_activities, embedding_dim)
        self.num_classes = num_classes
        
        # If activity classes are provided, add class embeddings
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, embedding_dim // 2)
            # Projection to combine activity and class embeddings
            self.combine_proj = nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim)
        
    def forward(self, activity_ids, activity_classes=None):
        act_emb = self.embedding(activity_ids)
        
        # If activity classes provided, concatenate class embeddings
        if self.num_classes > 0 and activity_classes is not None:
            class_emb = self.class_embedding(activity_classes)
            combined = torch.cat([act_emb, class_emb], dim=-1)
            return self.combine_proj(combined)
        
        return act_emb

class TwoTowerModel(nn.Module):
    """
    Two Towers Model for recommendations.
    Enhanced to include age, gender, and activity classes.
    Learns embeddings for users and activities separately, then computes similarity.
    """
    
    def __init__(self, user_feature_dim, num_activities, embedding_dim=64, num_classes=4):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(user_feature_dim, embedding_dim)
        self.activity_tower = ActivityTower(num_activities, embedding_dim, num_classes=num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
    def forward(self, user_features, activity_ids, activity_classes=None):
        """
        Forward pass.
        
        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim)
            activity_ids: Tensor of shape (batch_size,) with activity indices
            activity_classes: Optional tensor of shape (batch_size,) with activity class indices
            
        Returns:
            Similarity scores (dot product of embeddings)
        """
        user_emb = self.user_tower(user_features)
        activity_emb = self.activity_tower(activity_ids, activity_classes)
        
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
        
    def prepare_data(self, user_profiles, interaction_matrix, activity_to_idx, 
                    use_feature_engineering=True, feature_engineer=None, 
                    include_gender=True, include_classes=True):
        """
        Prepare data for training.
        Enhanced to include age, gender, and activity classes.
        
        Args:
            user_profiles: DataFrame with user features (may have missing values)
            interaction_matrix: DataFrame with user_id, activity, count/rating
            activity_to_idx: Mapping from activity name to index
            use_feature_engineering: Whether to use FeatureEngineer for handling missing data
            feature_engineer: Optional FeatureEngineer instance (creates new if None)
            include_gender: Whether to include gender features
            include_classes: Whether to include activity class features
        """
        import pandas as pd
        import numpy as np
        from data_generator import ACTIVITY_TO_CLASS
        
        # Merge user profiles with interactions
        # Use enhanced features: age, weight, height, gender, activity classes
        base_cols = ['user_id', 'age', 'weight_kg']
        if 'height_cm' in user_profiles.columns:
            base_cols.append('height_cm')
        if include_gender and 'gender' in user_profiles.columns:
            base_cols.append('gender')
        
        data = interaction_matrix.merge(
            user_profiles[base_cols],
            on='user_id',
            how='left'
        )
        
        # Add activity class if available
        if include_classes and 'activity' in data.columns:
            data['activity_class'] = data['activity'].map(ACTIVITY_TO_CLASS)
            # Encode activity classes
            class_to_idx = {'cardio': 0, 'strength': 1, 'flexibility': 2, 'sports': 3}
            data['activity_class_idx'] = data['activity_class'].map(class_to_idx).fillna(0)
        
        # Handle missing values
        if use_feature_engineering:
            if feature_engineer is None:
                from feature_engineering import FeatureEngineer
                feature_engineer = FeatureEngineer()
            
            # Impute missing values
            for col in base_cols:
                if col != 'user_id' and data[col].isna().any():
                    if col == 'height_cm':
                        fill_value = data[col].mean() if data[col].notna().any() else 170
                    elif col == 'age':
                        fill_value = data[col].mean() if data[col].notna().any() else 35
                    elif col == 'weight_kg':
                        fill_value = data[col].mean() if data[col].notna().any() else 70
                    elif col == 'gender':
                        fill_value = 'M'  # Default
                    else:
                        fill_value = 0
                    data[col] = data[col].fillna(fill_value)
        else:
            # Simple imputation
            data['height_cm'] = data['height_cm'].fillna(data['height_cm'].mean() if data['height_cm'].notna().any() else 170)
            data['age'] = data['age'].fillna(data['age'].mean() if data['age'].notna().any() else 35)
            data['weight_kg'] = data['weight_kg'].fillna(data['weight_kg'].mean() if data['weight_kg'].notna().any() else 70)
            if 'gender' in data.columns:
                data['gender'] = data['gender'].fillna('M')
        
        # Create activity indices
        data['activity_idx'] = data['activity'].map(activity_to_idx)
        data = data.dropna(subset=['activity_idx'])  # Remove any unmapped activities
        
        # Prepare user features: age, weight, height, gender (encoded), activity class preferences
        feature_list = []
        
        # Basic features
        for col in ['age', 'weight_kg', 'height_cm']:
            if col in data.columns:
                feature_list.append(data[col].values)
        
        # Gender encoding: M=1, F=0, Other=0.5
        if include_gender and 'gender' in data.columns:
            gender_encoded = data['gender'].map({'M': 1.0, 'F': 0.0, 'Other': 0.5}).fillna(0.5)
            feature_list.append(gender_encoded.values)
        
        # Activity class preferences (from user's favorite activities if available)
        if include_classes:
            # Get user's class preferences from favorite activities
            user_class_prefs = {}
            for user_id in data['user_id'].unique():
                user_row = user_profiles[user_profiles['user_id'] == user_id]
                if len(user_row) > 0 and 'favorite_activities' in user_row.columns:
                    fav_activities = str(user_row.iloc[0].get('favorite_activities', ''))
                    if pd.notna(fav_activities) and fav_activities:
                        fav_list = fav_activities.split(',')
                        class_counts = {'cardio': 0, 'strength': 0, 'flexibility': 0, 'sports': 0}
                        for act in fav_list:
                            act_class = ACTIVITY_TO_CLASS.get(act.strip(), 'cardio')
                            if act_class in class_counts:
                                class_counts[act_class] += 1
                        total = sum(class_counts.values()) if sum(class_counts.values()) > 0 else 1
                        user_class_prefs[user_id] = [
                            class_counts['cardio']/total,
                            class_counts['strength']/total,
                            class_counts['flexibility']/total,
                            class_counts['sports']/total
                        ]
                    else:
                        user_class_prefs[user_id] = [0.25, 0.25, 0.25, 0.25]
                else:
                    user_class_prefs[user_id] = [0.25, 0.25, 0.25, 0.25]
            
            # Add class preferences to features
            class_prefs = np.array([user_class_prefs[uid] for uid in data['user_id']])
            for i in range(4):
                feature_list.append(class_prefs[:, i])
        
        # Combine all features
        user_features = np.column_stack(feature_list) if feature_list else np.array([])
        
        # Scale features
        if len(user_features) > 0:
            user_features = self.scaler.fit_transform(user_features)
        
        activity_indices = data['activity_idx'].values
        activity_classes = data['activity_class_idx'].values if include_classes and 'activity_class_idx' in data.columns else None
        
        labels = data['count' if 'count' in data.columns else 'avg_rating'].values
        
        # Normalize labels to [0, 1] for binary cross-entropy or use as-is for MSE
        if len(labels) > 0 and labels.max() > 1:
            labels = labels / labels.max()
        
        # Return activity_classes only if include_classes is True
        if include_classes and activity_classes is not None:
            return user_features, activity_indices, labels, activity_classes
        else:
            return user_features, activity_indices, labels
    
    def train(self, user_features, activity_indices, labels, 
              epochs=50, batch_size=64, lr=0.001, val_split=0.2, activity_classes=None):
        """Train the model."""
        # Split data
        if activity_classes is not None:
            X_user_train, X_user_val, X_act_train, X_act_val, y_train, y_val, X_class_train, X_class_val = train_test_split(
                user_features, activity_indices, labels, activity_classes,
                test_size=val_split, random_state=42
            )
        else:
            X_user_train, X_user_val, X_act_train, X_act_val, y_train, y_val = train_test_split(
                user_features, activity_indices, labels,
                test_size=val_split, random_state=42
            )
            X_class_train, X_class_val = None, None
        
        # Convert to tensors
        X_user_train = torch.FloatTensor(X_user_train).to(self.device)
        X_user_val = torch.FloatTensor(X_user_val).to(self.device)
        X_act_train = torch.LongTensor(X_act_train).to(self.device)
        X_act_val = torch.LongTensor(X_act_val).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        if activity_classes is not None:
            X_class_train = torch.LongTensor(X_class_train).to(self.device)
            X_class_val = torch.LongTensor(X_class_val).to(self.device)
        
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
                
                if activity_classes is not None:
                    batch_class = X_class_train[batch_indices]
                    predictions = self.model(batch_user, batch_act, batch_class)
                else:
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
                if activity_classes is not None:
                    val_predictions = self.model(X_user_val, X_act_val, X_class_val)
                else:
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
        Handles both complete and incomplete user features.
        
        Args:
            user_features: Array of shape (user_feature_dim,) - [age, weight, height] or dict/list
                          Can be minimal (age, weight) or full (age, weight, height)
            activity_to_idx: Mapping from activity name to index
            top_k: Number of recommendations
            exclude_activities: Set of activity names to exclude
            
        Returns:
            List of tuples (activity, score)
        """
        import numpy as np
        
        self.model.eval()
        
        # Handle different input formats
        if isinstance(user_features, dict):
            # Extract features from dict
            feature_list = []
            for col in ['age', 'weight_kg', 'height_cm']:
                if col in user_features:
                    feature_list.append(user_features[col])
                elif col == 'height_cm':
                    # Use default height if missing
                    feature_list.append(170)  # Default height
                else:
                    feature_list.append(0)
            user_features = np.array(feature_list)
        elif isinstance(user_features, list):
            user_features = np.array(user_features)
        
        # Ensure we have the right number of features
        # If we only have age and weight, pad with default height
        if len(user_features) == 2:
            user_features = np.append(user_features, 170)  # Default height
        
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

