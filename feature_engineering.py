"""
Feature Engineering Utilities for Recommendation System
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

class FeatureEngineer:
    """Utilities for feature engineering."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.pca = None
        
    def create_user_features(self, user_profiles):
        """
        Create engineered features from user profiles.
        
        Args:
            user_profiles: DataFrame with age, weight_kg, height_cm, gender
            
        Returns:
            DataFrame with additional features
        """
        df = user_profiles.copy()
        
        # BMI (Body Mass Index)
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # One-hot encode categorical features
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
        age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        bmi_category_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_category')
        
        # Combine all features
        feature_df = pd.concat([
            df[['user_id', 'age', 'weight_kg', 'height_cm', 'bmi']],
            gender_dummies,
            age_group_dummies,
            bmi_category_dummies
        ], axis=1)
        
        return feature_df
    
    def create_activity_features(self, activity_sequences):
        """
        Create features from activity sequences.
        
        Args:
            activity_sequences: DataFrame with user_id, activity, date, duration_minutes, rating
            
        Returns:
            DataFrame with activity-level features per user
        """
        df = activity_sequences.copy()
        
        # User-level activity statistics
        user_stats = df.groupby('user_id').agg({
            'activity': 'count',  # Total activities
            'duration_minutes': ['mean', 'sum'],  # Avg and total duration
            'rating': 'mean',  # Average rating
            'date': ['min', 'max']  # First and last activity dates
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = [
            'user_id', 'total_activities', 'avg_duration', 'total_duration',
            'avg_rating', 'first_activity_date', 'last_activity_date'
        ]
        
        # Activity frequency per user
        activity_freq = df.groupby(['user_id', 'activity']).size().reset_index(name='frequency')
        activity_freq_pivot = activity_freq.pivot(
            index='user_id',
            columns='activity',
            values='frequency'
        ).fillna(0)
        
        # Most frequent activity per user
        user_stats['most_frequent_activity'] = df.groupby('user_id')['activity'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        ).values
        
        # Activity diversity (number of unique activities)
        user_stats['activity_diversity'] = df.groupby('user_id')['activity'].nunique().values
        
        # Recent activity trend (activities in last 30 days vs previous 30 days)
        df['date'] = pd.to_datetime(df['date'])
        recent_cutoff = df['date'].max() - pd.Timedelta(days=30)
        recent_activities = df[df['date'] >= recent_cutoff].groupby('user_id').size()
        previous_activities = df[df['date'] < recent_cutoff].groupby('user_id').size()
        
        user_stats['recent_activity_count'] = user_stats['user_id'].map(recent_activities).fillna(0)
        user_stats['previous_activity_count'] = user_stats['user_id'].map(previous_activities).fillna(0)
        user_stats['activity_trend'] = (
            user_stats['recent_activity_count'] - user_stats['previous_activity_count']
        )
        
        return user_stats
    
    def normalize_features(self, features, method='standard', feature_cols=None):
        """
        Normalize features.
        
        Args:
            features: DataFrame or array of features
            method: 'standard' (z-score) or 'minmax' (0-1 scaling)
            feature_cols: List of column names to normalize (if DataFrame)
            
        Returns:
            Normalized features
        """
        if isinstance(features, pd.DataFrame):
            if feature_cols is None:
                feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            
            scaler_key = f"{method}_{'_'.join(feature_cols)}"
            
            if scaler_key not in self.scalers:
                if method == 'standard':
                    self.scalers[scaler_key] = StandardScaler()
                else:
                    self.scalers[scaler_key] = MinMaxScaler()
                
                normalized_values = self.scalers[scaler_key].fit_transform(features[feature_cols])
            else:
                normalized_values = self.scalers[scaler_key].transform(features[feature_cols])
            
            features_normalized = features.copy()
            features_normalized[feature_cols] = normalized_values
            
            return features_normalized
        else:
            # NumPy array
            scaler_key = f"{method}_array"
            if scaler_key not in self.scalers:
                if method == 'standard':
                    self.scalers[scaler_key] = StandardScaler()
                else:
                    self.scalers[scaler_key] = MinMaxScaler()
                return self.scalers[scaler_key].fit_transform(features)
            else:
                return self.scalers[scaler_key].transform(features)
    
    def encode_categorical(self, data, columns):
        """
        Label encode categorical columns.
        
        Args:
            data: DataFrame
            columns: List of column names to encode
            
        Returns:
            DataFrame with encoded columns
        """
        encoded_data = data.copy()
        
        for col in columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_data[col] = self.encoders[col].fit_transform(data[col])
            else:
                # Handle unseen categories
                unique_values = set(data[col].unique())
                known_values = set(self.encoders[col].classes_)
                if unique_values - known_values:
                    # Add unknown category
                    all_values = list(known_values) + list(unique_values - known_values)
                    self.encoders[col].classes_ = np.array(all_values)
                
                encoded_data[col] = encoded_data[col].map(
                    {val: idx for idx, val in enumerate(self.encoders[col].classes_)}
                ).fillna(len(self.encoders[col].classes_) - 1)
        
        return encoded_data
    
    def apply_pca(self, features, n_components=10):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            features: DataFrame or array of features
            n_components: Number of principal components
            
        Returns:
            Transformed features
        """
        if isinstance(features, pd.DataFrame):
            feature_array = features.select_dtypes(include=[np.number]).values
        else:
            feature_array = features
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            transformed = self.pca.fit_transform(feature_array)
        else:
            transformed = self.pca.transform(feature_array)
        
        if isinstance(features, pd.DataFrame):
            pca_df = pd.DataFrame(
                transformed,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=features.index
            )
            return pca_df
        else:
            return transformed

