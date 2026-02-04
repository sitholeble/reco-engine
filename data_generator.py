"""
Generate fake data for recommendation system.
Creates user profiles and activity sequences.
Handles two user types:
- Long-standing members: robust data with full metadata
- New users: skeletal data (age, weight, height)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Activity types
ACTIVITIES = [
    'spin_class', 'weight_training', 'cycling', 'yoga', 'pilates',
    'running', 'swimming', 'boxing', 'dance', 'crossfit',
    'hiit', 'stretching', 'walking', 'tennis', 'basketball'
]

# Activity classes/categories
ACTIVITY_CLASSES = {
    'cardio': ['spin_class', 'cycling', 'running', 'hiit', 'swimming', 'walking', 'dance'],
    'strength': ['weight_training', 'crossfit', 'boxing'],
    'flexibility': ['yoga', 'pilates', 'stretching'],
    'sports': ['tennis', 'basketball']
}

# Reverse mapping: activity -> class
ACTIVITY_TO_CLASS = {}
for activity_class, activities in ACTIVITY_CLASSES.items():
    for activity in activities:
        ACTIVITY_TO_CLASS[activity] = activity_class

def get_cardio_activities():
    """Get list of cardio activities."""
    return ACTIVITY_CLASSES.get('cardio', [])

# Fitness goals
FITNESS_GOALS = [
    'weight_loss', 'muscle_gain', 'endurance', 'flexibility', 
    'general_fitness', 'rehabilitation', 'sports_performance'
]

# Time preferences
TIME_PREFERENCES = ['morning', 'afternoon', 'evening', 'flexible']

# Equipment preferences
EQUIPMENT_PREFERENCES = ['minimal', 'full_gym', 'cardio_focused', 'weights_focused', 'mixed']

def generate_user_profiles(n_users=1000, new_user_ratio=0.3, seed=42):
    """
    Generate fake user profiles with two types:
    - Long-standing members: robust data with full metadata
    - New users: skeletal data (age, weight, height)
    
    Args:
        n_users: Total number of users to generate
        new_user_ratio: Proportion of users that are new (default 0.3 = 30%)
        seed: Random seed for reproducibility (default 42)
    
    Returns:
        DataFrame with user profiles
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_new_users = int(n_users * new_user_ratio)
    n_long_standing = n_users - n_new_users
    
    # Common fields for all users
    all_user_ids = list(range(1, n_users + 1))
    all_ages = np.random.normal(35, 10, n_users).astype(int).clip(18, 70)
    all_weights = np.random.normal(70, 15, n_users).clip(45, 150)
    all_heights = np.random.normal(170, 10, n_users).clip(150, 200)
    all_genders = np.random.choice(['M', 'F', 'Other'], n_users, p=[0.45, 0.45, 0.1])
    
    # Determine which users are new vs long-standing
    new_user_ids = set(random.sample(all_user_ids, n_new_users))
    
    # Initialize data structure
    profiles = []
    
    for i, user_id in enumerate(all_user_ids):
        is_new_user = user_id in new_user_ids
        membership_type = 'new_user' if is_new_user else 'long_standing'
        
        # Base profile (all users have this)
        profile = {
            'user_id': user_id,
            'age': all_ages[i],
            'weight_kg': all_weights[i],
            'height_cm': all_heights[i],
            'gender': all_genders[i],
            'membership_type': membership_type
        }
        
        if is_new_user:
            # New users: minimal data
            # Only age, weight, height are guaranteed
            # Height might be missing for some new users
            if random.random() < 0.2:  # 20% of new users missing height
                profile['height_cm'] = None
            
            # All other fields are None for new users
            profile.update({
                'membership_duration_days': None,
                'total_classes_attended': None,
                'fitness_goal': None,
                'preferred_time_of_day': None,
                'equipment_preference': None,
                'favorite_activities': None,
                'injury_history': None,
                'activity_level': None,
                'avg_rating_given': None,
                'preferred_duration_minutes': None,
                'membership_tier': None,
                'has_trainer': None,
                'nutrition_tracking': None
            })
        else:
            # Long-standing members: robust metadata
            membership_days = np.random.randint(30, 730)  # 1 month to 2 years
            total_classes = np.random.randint(10, 200)
            
            # Generate favorite activities (2-4 activities)
            n_favorites = np.random.randint(2, 5)
            favorites = random.sample(ACTIVITIES, n_favorites)
            
            profile.update({
                'membership_duration_days': membership_days,
                'total_classes_attended': total_classes,
                'fitness_goal': np.random.choice(FITNESS_GOALS, p=[0.25, 0.20, 0.15, 0.10, 0.15, 0.05, 0.10]),
                'preferred_time_of_day': np.random.choice(TIME_PREFERENCES, p=[0.3, 0.25, 0.35, 0.1]),
                'equipment_preference': np.random.choice(EQUIPMENT_PREFERENCES, p=[0.1, 0.3, 0.2, 0.2, 0.2]),
                'favorite_activities': ','.join(favorites),
                'injury_history': np.random.choice(['none', 'knee', 'back', 'shoulder', 'ankle'], 
                                                   p=[0.6, 0.15, 0.10, 0.10, 0.05]),
                'activity_level': np.random.choice(['beginner', 'intermediate', 'advanced'], 
                                                   p=[0.2, 0.5, 0.3]),
                'avg_rating_given': np.clip(np.random.normal(4.2, 0.5), 1, 5),
                'preferred_duration_minutes': np.random.choice([30, 45, 60, 90], p=[0.1, 0.3, 0.5, 0.1]),
                'membership_tier': np.random.choice(['basic', 'premium', 'elite'], p=[0.5, 0.35, 0.15]),
                'has_trainer': np.random.choice([True, False], p=[0.3, 0.7]),
                'nutrition_tracking': np.random.choice([True, False], p=[0.4, 0.6])
            })
        
        profiles.append(profile)
    
    df = pd.DataFrame(profiles)
    return df

def generate_activity_sequences(user_profiles, min_activities_new=0, max_activities_new=3, 
                                min_activities_long=10, max_activities_long=100, seed=42):
    """
    Generate activity sequences for users.
    New users have few/no activities, long-standing members have many.
    
    Args:
        user_profiles: DataFrame with user profiles (must have membership_type)
        min_activities_new: Minimum activities for new users
        max_activities_new: Maximum activities for new users
        min_activities_long: Minimum activities for long-standing members
        max_activities_long: Maximum activities for long-standing members
        seed: Random seed for reproducibility (default 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    sequences = []
    
    # Create a mapping of user_id to membership type
    user_types = dict(zip(user_profiles['user_id'], user_profiles['membership_type']))
    
    # For long-standing members, get their favorite activities if available
    long_standing_favorites = {}
    for _, row in user_profiles[user_profiles['membership_type'] == 'long_standing'].iterrows():
        if pd.notna(row.get('favorite_activities')):
            long_standing_favorites[row['user_id']] = row['favorite_activities'].split(',')
    
    for user_id in user_profiles['user_id']:
        membership_type = user_types[user_id]
        
        # Determine number of activities based on user type
        if membership_type == 'new_user':
            # New users: 0-3 activities (many have none)
            n_activities = np.random.choice(
                [0, 1, 2, 3],
                p=[0.4, 0.3, 0.2, 0.1]  # 40% have no activities yet
            )
            # Shorter history window for new users
            start_date = datetime.now() - timedelta(days=np.random.randint(7, 30))
        else:
            # Long-standing members: 10-100 activities
            n_activities = np.random.randint(min_activities_long, max_activities_long + 1)
            # Longer history window
            membership_days = user_profiles[user_profiles['user_id'] == user_id]['membership_duration_days'].iloc[0]
            if pd.notna(membership_days):
                start_date = datetime.now() - timedelta(days=int(membership_days))
            else:
                start_date = datetime.now() - timedelta(days=180)
        
        # Get user's favorite activities if available
        user_favorites = long_standing_favorites.get(user_id, [])
        
        # Create activity sequence with temporal patterns
        for i in range(n_activities):
            activity_date = start_date + timedelta(days=i * np.random.uniform(0.5, 3))
            
            # Create patterns: users tend to repeat activities they like
            if i == 0:
                if user_favorites:
                    # Long-standing members start with favorite activities
                    activity = random.choice(user_favorites)
                else:
                    activity = np.random.choice(ACTIVITIES)
            else:
                # 50% chance to repeat last activity, 30% chance for favorite, 20% for new
                rand_val = random.random()
                if rand_val < 0.5 and len(sequences) > 0:
                    # Repeat last activity
                    activity = sequences[-1]['activity']
                elif rand_val < 0.8 and user_favorites:
                    # Use favorite activity
                    activity = random.choice(user_favorites)
                else:
                    activity = np.random.choice(ACTIVITIES)
            
            # Duration based on user preference if available
            user_row = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
            if pd.notna(user_row.get('preferred_duration_minutes')):
                preferred_duration = user_row['preferred_duration_minutes']
                duration = int(np.clip(np.random.normal(preferred_duration, 15), 30, 120))
            else:
                duration = np.random.randint(30, 120)
            
            # Rating: long-standing members tend to rate higher (they're engaged)
            if membership_type == 'long_standing':
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.05, 0.13, 0.35, 0.45])
            else:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.25, 0.3, 0.2])
            
            # Get activity class
            activity_class = ACTIVITY_TO_CLASS.get(activity, 'other')
            
            sequences.append({
                'user_id': user_id,
                'activity': activity,
                'activity_class': activity_class,
                'date': activity_date.strftime('%Y-%m-%d'),
                'duration_minutes': duration,
                'rating': rating
            })
    
    df = pd.DataFrame(sequences)
    if len(df) > 0:
        return df.sort_values(['user_id', 'date'])
    else:
        return df

def generate_interaction_matrix(user_profiles, activity_sequences):
    """
    Generate user-activity interaction matrix for collaborative filtering.
    Handles cases where users may have no activities (new users).
    """
    if len(activity_sequences) == 0:
        # Return empty matrix with proper structure
        return pd.DataFrame(columns=['user_id', 'activity', 'count', 'avg_rating'])
    
    # Create user-activity matrix (count of interactions)
    interaction_matrix = activity_sequences.groupby(['user_id', 'activity']).size().reset_index(name='count')
    
    # Add ratings (average rating per user-activity pair)
    ratings = activity_sequences.groupby(['user_id', 'activity'])['rating'].mean().reset_index(name='avg_rating')
    interaction_matrix = interaction_matrix.merge(ratings, on=['user_id', 'activity'], how='left')
    
    return interaction_matrix

if __name__ == '__main__':
    import os
    
    print("Generating fake data...")
    print("="*60)
    
    # Generate user profiles (30% new users, 70% long-standing)
    # Using fixed seed=42 to lock the dataset
    user_profiles = generate_user_profiles(n_users=1000, new_user_ratio=0.3, seed=42)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    user_profiles.to_csv('data/user_profiles.csv', index=False)
    
    n_new = len(user_profiles[user_profiles['membership_type'] == 'new_user'])
    n_long = len(user_profiles[user_profiles['membership_type'] == 'long_standing'])
    print(f"Generated {len(user_profiles)} user profiles:")
    print(f"  - New users: {n_new} (skeletal data)")
    print(f"  - Long-standing members: {n_long} (robust metadata)")
    
    # Generate activity sequences (based on user profiles)
    # Using fixed seed=42 to lock the dataset
    activity_sequences = generate_activity_sequences(user_profiles, seed=42)
    activity_sequences.to_csv('data/activity_sequences.csv', index=False)
    
    n_activities_new = len(activity_sequences[activity_sequences['user_id'].isin(
        user_profiles[user_profiles['membership_type'] == 'new_user']['user_id']
    )])
    n_activities_long = len(activity_sequences[activity_sequences['user_id'].isin(
        user_profiles[user_profiles['membership_type'] == 'long_standing']['user_id']
    )])
    
    print(f"\nGenerated {len(activity_sequences)} activity records:")
    print(f"  - From new users: {n_activities_new}")
    print(f"  - From long-standing members: {n_activities_long}")
    
    # Generate interaction matrix
    interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
    interaction_matrix.to_csv('data/interaction_matrix.csv', index=False)
    print(f"\nGenerated interaction matrix with {len(interaction_matrix)} user-activity pairs")
    
    print("\n" + "="*60)
    print("Sample user profiles (new user):")
    print("="*60)
    new_user_sample = user_profiles[user_profiles['membership_type'] == 'new_user'].head(2)
    print(new_user_sample[['user_id', 'age', 'weight_kg', 'height_cm', 'membership_type']].to_string())
    
    print("\n" + "="*60)
    print("Sample user profiles (long-standing member):")
    print("="*60)
    long_sample = user_profiles[user_profiles['membership_type'] == 'long_standing'].head(2)
    print(long_sample[['user_id', 'age', 'weight_kg', 'height_cm', 'membership_type', 
                       'membership_duration_days', 'total_classes_attended', 
                       'fitness_goal', 'favorite_activities']].to_string())
    
    print("\n" + "="*60)
    print("Sample activity sequences:")
    print("="*60)
    if len(activity_sequences) > 0:
        print(activity_sequences.head(10).to_string())
    else:
        print("No activity sequences (all new users have no activities)")
    
    print("\n" + "="*60)
    print("Sample interaction matrix:")
    print("="*60)
    if len(interaction_matrix) > 0:
        print(interaction_matrix.head().to_string())
    else:
        print("No interactions (all new users)")

