"""
Feature Engineering Utilities for Recommendation System
Handles both complete (long-standing members) and incomplete (new users) data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """
    Utilities for feature engineering.
    Handles missing data gracefully for new users with skeletal data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.pca = None
        self.imputers = {}
        self.default_values = {}
        
    def impute_missing_values(self, df, strategy='mean', categorical_strategy='most_frequent'):
        """
        Impute missing values in DataFrame.
        Stores imputation strategy for later use on new data.
        
        Args:
            df: DataFrame with potentially missing values
            strategy: Strategy for numerical columns ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
        
        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute numerical columns
        for col in numerical_cols:
            if df[col].isna().any():
                if col not in self.imputers:
                    if strategy == 'mean':
                        fill_value = df[col].mean()
                    elif strategy == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = 0
                    
                    self.imputers[col] = SimpleImputer(strategy=strategy, fill_value=fill_value)
                    df_imputed[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                    self.default_values[col] = fill_value
                else:
                    df_imputed[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # Impute categorical columns
        for col in categorical_cols:
            if df[col].isna().any():
                if col not in self.imputers:
                    most_frequent = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                    self.imputers[col] = SimpleImputer(strategy=categorical_strategy, fill_value=most_frequent)
                    df_imputed[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                    self.default_values[col] = most_frequent
                else:
                    df_imputed[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        return df_imputed
    
    def create_user_features(self, user_profiles, handle_missing=True):
        """
        Create engineered features from user profiles.
        Handles missing data for new users gracefully.
        
        Args:
            user_profiles: DataFrame with user profiles (may have missing values)
            handle_missing: Whether to impute missing values (default True)
            
        Returns:
            DataFrame with additional features
        """
        df = user_profiles.copy()
        
        # Handle missing height (some new users may not have it)
        if handle_missing and 'height_cm' in df.columns:
            # Impute missing height with mean or median
            if df['height_cm'].isna().any():
                if 'height_cm' not in self.default_values:
                    # Use mean height from available data
                    self.default_values['height_cm'] = df['height_cm'].mean()
                df['height_cm'] = df['height_cm'].fillna(self.default_values['height_cm'])
        
        # BMI (Body Mass Index) - handle missing height
        df['bmi'] = df.apply(
            lambda row: row['weight_kg'] / ((row['height_cm'] / 100) ** 2) 
            if pd.notna(row['height_cm']) and pd.notna(row['weight_kg']) 
            else None,
            axis=1
        )
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # BMI categories (handle missing BMI)
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        ) if df['bmi'].notna().any() else pd.Series(['Normal'] * len(df), index=df.index)
        
        # Impute missing values if requested
        if handle_missing:
            df = self.impute_missing_values(df)
        
        # One-hot encode categorical features (handle missing values)
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender', dummy_na=False)
        age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group', dummy_na=False)
        bmi_category_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_category', dummy_na=False)
        
        # Add membership type features if available
        membership_features = pd.DataFrame()
        if 'membership_type' in df.columns:
            membership_features = pd.get_dummies(df['membership_type'], prefix='membership_type', dummy_na=False)
        
        # Add long-standing member metadata features
        metadata_features = pd.DataFrame()
        if 'membership_duration_days' in df.columns:
            # Normalize membership duration
            if df['membership_duration_days'].notna().any():
                max_duration = df['membership_duration_days'].max()
                if pd.notna(max_duration) and max_duration > 0:
                    metadata_features['membership_duration_normalized'] = (
                        df['membership_duration_days'].fillna(0) / max_duration
                    )
                else:
                    metadata_features['membership_duration_normalized'] = 0
        
        if 'total_classes_attended' in df.columns:
            if df['total_classes_attended'].notna().any():
                max_classes = df['total_classes_attended'].max()
                if pd.notna(max_classes) and max_classes > 0:
                    metadata_features['total_classes_normalized'] = (
                        df['total_classes_attended'].fillna(0) / max_classes
                    )
                else:
                    metadata_features['total_classes_normalized'] = 0
        
        # Combine all features
        base_features = ['user_id', 'age', 'weight_kg']
        if 'height_cm' in df.columns:
            base_features.append('height_cm')
        if 'bmi' in df.columns:
            base_features.append('bmi')
        
        feature_list = [df[base_features]]
        
        if len(gender_dummies.columns) > 0:
            feature_list.append(gender_dummies)
        if len(age_group_dummies.columns) > 0:
            feature_list.append(age_group_dummies)
        if len(bmi_category_dummies.columns) > 0:
            feature_list.append(bmi_category_dummies)
        if len(membership_features.columns) > 0:
            feature_list.append(membership_features)
        if len(metadata_features.columns) > 0:
            feature_list.append(metadata_features)
        
        feature_df = pd.concat(feature_list, axis=1)
        
        return feature_df
    
    def create_activity_features(self, activity_sequences, user_profiles=None):
        """
        Create features from activity sequences.
        Handles users with no activities (new users) gracefully.
        
        Args:
            activity_sequences: DataFrame with user_id, activity, date, duration_minutes, rating
            user_profiles: Optional DataFrame with all user_ids (to include users with no activities)
            
        Returns:
            DataFrame with activity-level features per user
        """
        if len(activity_sequences) == 0:
            # Return empty stats with proper structure
            if user_profiles is not None:
                return pd.DataFrame({
                    'user_id': user_profiles['user_id'],
                    'total_activities': 0,
                    'avg_duration': 0,
                    'total_duration': 0,
                    'avg_rating': 0,
                    'activity_diversity': 0,
                    'recent_activity_count': 0,
                    'previous_activity_count': 0,
                    'activity_trend': 0
                })
            else:
                return pd.DataFrame(columns=['user_id', 'total_activities', 'avg_duration', 
                                            'total_duration', 'avg_rating', 'activity_diversity',
                                            'recent_activity_count', 'previous_activity_count', 'activity_trend'])
        
        df = activity_sequences.copy()
        
        # Get all user_ids (including those with no activities)
        if user_profiles is not None:
            all_user_ids = user_profiles['user_id'].unique()
        else:
            all_user_ids = df['user_id'].unique()
        
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
        
        # Add users with no activities (fill with zeros/defaults)
        users_with_activities = set(user_stats['user_id'].unique())
        users_without_activities = set(all_user_ids) - users_with_activities
        
        if users_without_activities:
            empty_stats = pd.DataFrame({
                'user_id': list(users_without_activities),
                'total_activities': 0,
                'avg_duration': 0,
                'total_duration': 0,
                'avg_rating': 0,
                'first_activity_date': None,
                'last_activity_date': None
            })
            user_stats = pd.concat([user_stats, empty_stats], ignore_index=True)
        
        # Most frequent activity per user (handle users with no activities)
        if len(df) > 0:
            most_frequent = df.groupby('user_id')['activity'].apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            )
            user_stats['most_frequent_activity'] = user_stats['user_id'].map(most_frequent)
        else:
            user_stats['most_frequent_activity'] = None
        
        # Activity diversity (number of unique activities)
        if len(df) > 0:
            diversity = df.groupby('user_id')['activity'].nunique()
            user_stats['activity_diversity'] = user_stats['user_id'].map(diversity).fillna(0)
        else:
            user_stats['activity_diversity'] = 0
        
        # Recent activity trend (activities in last 30 days vs previous 30 days)
        if len(df) > 0 and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].notna().any():
                recent_cutoff = df['date'].max() - pd.Timedelta(days=30)
                recent_activities = df[df['date'] >= recent_cutoff].groupby('user_id').size()
                previous_activities = df[df['date'] < recent_cutoff].groupby('user_id').size()
                
                user_stats['recent_activity_count'] = user_stats['user_id'].map(recent_activities).fillna(0)
                user_stats['previous_activity_count'] = user_stats['user_id'].map(previous_activities).fillna(0)
                user_stats['activity_trend'] = (
                    user_stats['recent_activity_count'] - user_stats['previous_activity_count']
                )
            else:
                user_stats['recent_activity_count'] = 0
                user_stats['previous_activity_count'] = 0
                user_stats['activity_trend'] = 0
        else:
            user_stats['recent_activity_count'] = 0
            user_stats['previous_activity_count'] = 0
            user_stats['activity_trend'] = 0
        
        # Ensure all users are included and sorted
        user_stats = user_stats.sort_values('user_id').reset_index(drop=True)
        
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
    
    def prepare_model_features(self, user_profiles, activity_sequences=None, 
                              use_minimal_for_new=True):
        """
        Prepare features for model training/prediction.
        Handles both new users (skeletal data) and long-standing members (robust data).
        
        Args:
            user_profiles: DataFrame with user profiles
            activity_sequences: Optional DataFrame with activity sequences
            use_minimal_for_new: If True, use only age/weight/height for new users
            
        Returns:
            Tuple of (feature_df, feature_columns) where feature_df is ready for models
        """
        # Create user features
        user_features = self.create_user_features(user_profiles, handle_missing=True)
        
        # Add activity features if available
        if activity_sequences is not None and len(activity_sequences) > 0:
            activity_features = self.create_activity_features(activity_sequences, user_profiles)
            # Merge activity features with user features
            user_features = user_features.merge(activity_features, on='user_id', how='left')
            # Fill missing activity features with 0
            activity_cols = ['total_activities', 'avg_duration', 'total_duration', 
                           'avg_rating', 'activity_diversity', 'recent_activity_count',
                           'previous_activity_count', 'activity_trend']
            for col in activity_cols:
                if col in user_features.columns:
                    user_features[col] = user_features[col].fillna(0)
        
        # For new users, optionally use only minimal features
        if use_minimal_for_new and 'membership_type' in user_profiles.columns:
            new_user_ids = user_profiles[user_profiles['membership_type'] == 'new_user']['user_id'].values
            # Keep all features but note which users are new
            user_features['is_new_user'] = user_features['user_id'].isin(new_user_ids).astype(int)
        
        # Select only numerical features for models (exclude user_id and categorical)
        feature_cols = [col for col in user_features.columns 
                       if col not in ['user_id', 'most_frequent_activity', 
                                     'first_activity_date', 'last_activity_date']
                       and user_features[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
        
        return user_features, feature_cols
    
    def get_minimal_features(self, user_profiles):
        """
        Extract minimal features for new users (age, weight, height only).
        Useful for cold-start recommendations.
        
        Args:
            user_profiles: DataFrame with user profiles
            
        Returns:
            DataFrame with minimal features (age, weight_kg, height_cm)
        """
        minimal_cols = ['user_id', 'age', 'weight_kg']
        if 'height_cm' in user_profiles.columns:
            minimal_cols.append('height_cm')
        
        minimal_features = user_profiles[minimal_cols].copy()
        
        # Fill missing height with default if available
        if 'height_cm' in minimal_features.columns and 'height_cm' in self.default_values:
            minimal_features['height_cm'] = minimal_features['height_cm'].fillna(
                self.default_values['height_cm']
            )
        
        return minimal_features

