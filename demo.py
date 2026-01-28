"""
Demo script for recommendation system.
Shows all three approaches: Collaborative Filtering, Two Towers, and Sequence Models.
"""
import os
import pandas as pd
import numpy as np
import torch

# Import models
from models.collaborative_filtering import CollaborativeFiltering
from models.two_towers import TwoTowerModel, TwoTowerTrainer
from models.sequence_model import ActivityLSTM, SequenceModelTrainer
from feature_engineering import FeatureEngineer

def ensure_data_exists():
    """Generate data if it doesn't exist."""
    if not os.path.exists('data/user_profiles.csv'):
        print("Generating data...")
        from data_generator import generate_user_profiles, generate_activity_sequences, generate_interaction_matrix
        
        user_profiles = generate_user_profiles(n_users=1000)
        os.makedirs('data', exist_ok=True)
        user_profiles.to_csv('data/user_profiles.csv', index=False)
        
        activity_sequences = generate_activity_sequences(n_users=1000)
        activity_sequences.to_csv('data/activity_sequences.csv', index=False)
        
        interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
        interaction_matrix.to_csv('data/interaction_matrix.csv', index=False)
        
        print("Data generated successfully!")

def demo_collaborative_filtering():
    """Demo collaborative filtering model."""
    print("\n" + "="*60)
    print("DEMO 1: Collaborative Filtering (Amazon-style)")
    print("="*60)
    
    # Load data
    interaction_matrix = pd.read_csv('data/interaction_matrix.csv')
    
    # Train model
    cf_model = CollaborativeFiltering()
    cf_model.fit(interaction_matrix)
    
    # Example 1: Find similar activities
    print("\n1. 'People who did spin_class also did...'")
    similar = cf_model.recommend_similar_activities('spin_class', top_k=5)
    for activity, score in similar:
        print(f"   - {activity}: similarity = {score:.4f}")
    
    # Example 2: Recommend for a user
    print("\n2. Recommendations for user_id=1:")
    user_recs = cf_model.recommend_for_user(user_id=1, top_k=5)
    for activity, score in user_recs:
        print(f"   - {activity}: score = {score:.4f}")
    
    # Example 3: Popular activities
    print("\n3. Most popular activities:")
    popular = cf_model.get_popular_activities(top_k=5)
    for activity, count in popular:
        print(f"   - {activity}: {int(count)} users")

def demo_two_towers():
    """Demo Two Towers model."""
    print("\n" + "="*60)
    print("DEMO 2: Two Towers Model")
    print("="*60)
    
    # Load data
    user_profiles = pd.read_csv('data/user_profiles.csv')
    interaction_matrix = pd.read_csv('data/interaction_matrix.csv')
    
    # Create activity mapping
    activities = sorted(interaction_matrix['activity'].unique())
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    
    # Initialize model
    user_feature_dim = 3  # age, weight, height
    num_activities = len(activities)
    model = TwoTowerModel(user_feature_dim, num_activities, embedding_dim=64)
    
    # Train
    trainer = TwoTowerTrainer(model, device='cpu')
    print("\nTraining Two Towers model...")
    user_features, activity_indices, labels = trainer.prepare_data(
        user_profiles, interaction_matrix, activity_to_idx
    )
    
    train_losses, val_losses = trainer.train(
        user_features, activity_indices, labels,
        epochs=30, batch_size=64, lr=0.001
    )
    
    # Example: Recommend for a specific user
    print("\nRecommendations for user (age=30, weight=70kg, height=175cm):")
    user_features_example = [30, 70, 175]
    recommendations = trainer.recommend(
        user_features_example,
        activity_to_idx,
        top_k=5
    )
    for activity, score in recommendations:
        print(f"   - {activity}: score = {score:.4f}")
    
    # Example: Get user's done activities and recommend
    user_id = 1
    user_done = interaction_matrix[interaction_matrix['user_id'] == user_id]['activity'].tolist()
    user_info = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
    
    print(f"\nUser {user_id} (age={user_info['age']}, weight={user_info['weight_kg']:.1f}kg, height={user_info['height_cm']:.1f}cm)")
    print(f"Previously did: {', '.join(user_done[:5])}")
    
    recommendations = trainer.recommend(
        [user_info['age'], user_info['weight_kg'], user_info['height_cm']],
        activity_to_idx,
        top_k=5,
        exclude_activities=set(user_done)
    )
    print("Recommendations:")
    for activity, score in recommendations:
        print(f"   - {activity}: score = {score:.4f}")

def demo_sequence_model():
    """Demo sequence prediction model."""
    print("\n" + "="*60)
    print("DEMO 3: LSTM Sequence Model (Next Activity Prediction)")
    print("="*60)
    
    # Load data
    activity_sequences = pd.read_csv('data/activity_sequences.csv')
    
    # Prepare sequences first to get vocab size
    print("\nPreparing sequences...")
    temp_trainer = SequenceModelTrainer(None, device='cpu')
    X, y, activity_to_idx = temp_trainer.prepare_sequences(activity_sequences, sequence_length=5)
    
    # Create model with correct vocab size
    vocab_size = len(activity_to_idx)
    model = ActivityLSTM(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)
    trainer = SequenceModelTrainer(model, device='cpu')
    trainer.label_encoder = temp_trainer.label_encoder
    
    # Train
    print("\nTraining LSTM model...")
    train_losses, val_losses = trainer.train(
        X, y, epochs=30, batch_size=64, lr=0.001
    )
    
    # Example: Predict next activity
    print("\nExample predictions:")
    
    # Example 1: Cycling -> Spin -> Weight Training -> ?
    sequence1 = ['cycling', 'spin_class', 'weight_training']
    print(f"\nSequence: {' -> '.join(sequence1)}")
    predictions = trainer.predict_next(sequence1, top_k=5)
    print("Next activity predictions:")
    for activity, prob in predictions:
        print(f"   - {activity}: probability = {prob:.4f}")
    
    # Example 2: Predict full sequence
    sequence2 = ['yoga', 'pilates']
    print(f"\nStarting sequence: {' -> '.join(sequence2)}")
    predicted_sequence = trainer.predict_sequence(sequence2, length=5)
    print(f"Predicted sequence: {' -> '.join(predicted_sequence)}")
    
    # Example 3: Real user sequence
    user_id = 1
    user_activities = activity_sequences[activity_sequences['user_id'] == user_id].sort_values('date')
    if len(user_activities) >= 3:
        recent_activities = user_activities['activity'].tail(3).tolist()
        print(f"\nUser {user_id} recent activities: {' -> '.join(recent_activities)}")
        predictions = trainer.predict_next(recent_activities, top_k=5)
        print("What they might do next:")
        for activity, prob in predictions:
            print(f"   - {activity}: probability = {prob:.4f}")

def demo_feature_engineering():
    """Demo feature engineering."""
    print("\n" + "="*60)
    print("DEMO 4: Feature Engineering")
    print("="*60)
    
    # Load data
    user_profiles = pd.read_csv('data/user_profiles.csv')
    activity_sequences = pd.read_csv('data/activity_sequences.csv')
    
    # Create feature engineer
    fe = FeatureEngineer()
    
    # User features
    print("\n1. Engineered User Features:")
    user_features = fe.create_user_features(user_profiles)
    print(f"   Original features: {list(user_profiles.columns)}")
    print(f"   New features added: BMI, age_group, bmi_category, one-hot encodings")
    print(f"   Total features: {len(user_features.columns)}")
    print("\n   Sample engineered features:")
    print(user_features[['user_id', 'age', 'bmi', 'age_group', 'bmi_category']].head())
    
    # Activity features
    print("\n2. Engineered Activity Features:")
    activity_features = fe.create_activity_features(activity_sequences)
    print(f"   Features created: total_activities, avg_duration, activity_diversity, activity_trend, etc.")
    print("\n   Sample activity features:")
    print(activity_features.head())
    
    # Normalization
    print("\n3. Feature Normalization:")
    normalized = fe.normalize_features(
        user_features[['age', 'weight_kg', 'height_cm', 'bmi']],
        method='standard'
    )
    print("   Normalized features (z-score):")
    print(normalized.head())

def main():
    """Run all demos."""
    print("="*60)
    print("RECOMMENDATION SYSTEM DEMO")
    print("="*60)
    
    # Ensure data exists
    ensure_data_exists()
    
    # Run demos
    demo_collaborative_filtering()
    demo_two_towers()
    demo_sequence_model()
    demo_feature_engineering()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)

if __name__ == '__main__':
    main()

