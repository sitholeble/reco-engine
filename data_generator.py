"""
Generate fake data for recommendation system.
Creates user profiles and activity sequences.
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

def generate_user_profiles(n_users=1000):
    """Generate fake user profiles with age, weight, height."""
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'user_id': range(1, n_users + 1),
        'age': np.random.normal(35, 10, n_users).astype(int).clip(18, 70),
        'weight_kg': np.random.normal(70, 15, n_users).clip(45, 150),
        'height_cm': np.random.normal(170, 10, n_users).clip(150, 200),
        'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.45, 0.45, 0.1])
    }
    
    df = pd.DataFrame(data)
    return df

def generate_activity_sequences(n_users=1000, min_activities=5, max_activities=50):
    """Generate activity sequences for users."""
    np.random.seed(42)
    random.seed(42)
    
    sequences = []
    
    for user_id in range(1, n_users + 1):
        n_activities = np.random.randint(min_activities, max_activities + 1)
        start_date = datetime.now() - timedelta(days=90)
        
        # Create activity sequence with temporal patterns
        for i in range(n_activities):
            activity_date = start_date + timedelta(days=i * np.random.uniform(0.5, 3))
            
            # Create some patterns: users tend to repeat activities they like
            if i == 0:
                activity = np.random.choice(ACTIVITIES)
            else:
                # 40% chance to repeat last activity, 60% chance for new one
                if random.random() < 0.4 and len(sequences) > 0:
                    activity = sequences[-1]['activity']
                else:
                    activity = np.random.choice(ACTIVITIES)
            
            sequences.append({
                'user_id': user_id,
                'activity': activity,
                'date': activity_date.strftime('%Y-%m-%d'),
                'duration_minutes': np.random.randint(30, 120),
                'rating': np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.3, 0.4])
            })
    
    df = pd.DataFrame(sequences)
    return df.sort_values(['user_id', 'date'])

def generate_interaction_matrix(user_profiles, activity_sequences):
    """Generate user-activity interaction matrix for collaborative filtering."""
    # Create user-activity matrix (count of interactions)
    interaction_matrix = activity_sequences.groupby(['user_id', 'activity']).size().reset_index(name='count')
    
    # Add ratings (average rating per user-activity pair)
    ratings = activity_sequences.groupby(['user_id', 'activity'])['rating'].mean().reset_index(name='avg_rating')
    interaction_matrix = interaction_matrix.merge(ratings, on=['user_id', 'activity'], how='left')
    
    return interaction_matrix

if __name__ == '__main__':
    print("Generating fake data...")
    
    # Generate user profiles
    user_profiles = generate_user_profiles(n_users=1000)
    user_profiles.to_csv('data/user_profiles.csv', index=False)
    print(f"Generated {len(user_profiles)} user profiles")
    
    # Generate activity sequences
    activity_sequences = generate_activity_sequences(n_users=1000)
    activity_sequences.to_csv('data/activity_sequences.csv', index=False)
    print(f"Generated {len(activity_sequences)} activity records")
    
    # Generate interaction matrix
    interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
    interaction_matrix.to_csv('data/interaction_matrix.csv', index=False)
    print(f"Generated interaction matrix with {len(interaction_matrix)} user-activity pairs")
    
    print("\nSample user profiles:")
    print(user_profiles.head())
    print("\nSample activity sequences:")
    print(activity_sequences.head(10))
    print("\nSample interaction matrix:")
    print(interaction_matrix.head())

