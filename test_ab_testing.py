"""
Test script for A/B testing framework.
Demonstrates how to set up experiments and evaluate different recommendation strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ab_testing_service import ABTestingService
from models.two_towers import TwoTowerModel, TwoTowerTrainer
from models.sequence_model import ActivityLSTM, SequenceModelTrainer
from models.collaborative_filtering import CollaborativeFiltering
from data_generator import generate_user_profiles, generate_activity_sequences, generate_interaction_matrix
import torch

def setup_models():
    """Set up trained models for A/B testing."""
    print("Setting up models...")
    
    # Generate sample data
    user_profiles = generate_user_profiles(n_users=500, seed=42)
    activity_sequences = generate_activity_sequences(user_profiles, seed=42)
    interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
    
    # Prepare activity mapping
    activities = sorted(interaction_matrix['activity'].unique())
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    
    # Train Two Tower model
    print("Training Two Tower model...")
    user_feature_dim = 8
    num_activities = len(activities)
    two_tower_model = TwoTowerModel(user_feature_dim, num_activities, embedding_dim=64, num_classes=4)
    two_tower_trainer = TwoTowerTrainer(two_tower_model, device='cpu')
    
    result = two_tower_trainer.prepare_data(
        user_profiles, interaction_matrix, activity_to_idx,
        include_gender=True, include_classes=True
    )
    if len(result) == 4:
        user_features, activity_indices, labels, activity_classes = result
    else:
        user_features, activity_indices, labels = result
        activity_classes = None
    
    two_tower_trainer.train(
        user_features, activity_indices, labels,
        epochs=20, batch_size=64, lr=0.001,
        activity_classes=activity_classes
    )
    
    # Train Sequence model
    print("Training Sequence model...")
    temp_trainer = SequenceModelTrainer(None, device='cpu')
    seq_result = temp_trainer.prepare_sequences(
        activity_sequences, user_profiles, sequence_length=5, use_user_features=True
    )
    if len(seq_result) == 4:
        X, y, user_features_seq, activity_to_idx_seq = seq_result
    else:
        X, y, activity_to_idx_seq = seq_result
        user_features_seq = None
    
    vocab_size = len(activity_to_idx_seq)
    user_feature_dim_seq = user_features_seq.shape[1] if user_features_seq is not None else 0
    sequence_model = ActivityLSTM(vocab_size, embedding_dim=64, hidden_dim=128, 
                                num_layers=2, user_feature_dim=user_feature_dim_seq)
    sequence_trainer = SequenceModelTrainer(sequence_model, device='cpu')
    sequence_trainer.label_encoder = temp_trainer.label_encoder
    
    sequence_trainer.train(
        X, y, epochs=20, batch_size=64, lr=0.001,
        user_features=user_features_seq
    )
    
    # Train Collaborative Filtering
    print("Training Collaborative Filtering model...")
    cf_model = CollaborativeFiltering()
    cf_model.fit(interaction_matrix)
    
    # Get popular activities
    popular_activities = cf_model.get_popular_activities(top_k=20)
    
    return {
        'two_tower_trainer': two_tower_trainer,
        'sequence_trainer': sequence_trainer,
        'cf_model': cf_model,
        'activity_to_idx': activity_to_idx,
        'popular_activities': popular_activities,
        'user_profiles': user_profiles,
        'activity_sequences': activity_sequences
    }

def simulate_user_interactions(service: ABTestingService, experiment_id: str,
                               models_data: Dict, n_users: int = 100):
    """Simulate user interactions for A/B testing."""
    print(f"\nSimulating interactions for {n_users} users...")
    
    user_profiles = models_data['user_profiles']
    activity_sequences = models_data['activity_sequences']
    
    # Select random users
    selected_users = user_profiles.sample(min(n_users, len(user_profiles)))
    
    interactions_count = 0
    clicks_count = 0
    bookings_count = 0
    
    for _, user_row in selected_users.iterrows():
        user_id = user_row['user_id']
        
        # Get user's activity history
        user_history = activity_sequences[
            activity_sequences['user_id'] == user_id
        ]['activity'].tolist()
        
        # Prepare user features
        user_features = {
            'age': user_row['age'],
            'weight_kg': user_row['weight_kg'],
            'height_cm': user_row['height_cm'],
            'gender': user_row['gender']
        }
        
        # Get recommendations
        try:
            recommendations, variant = service.get_recommendations(
                user_id=user_id,
                experiment_id=experiment_id,
                user_features=user_features,
                activity_history=user_history,
                top_k=5
            )
            interactions_count += 1
            
            # Simulate click (30% probability)
            if recommendations and np.random.random() < 0.3:
                clicked_activity, _ = np.random.choice(
                    [r[0] for r in recommendations],
                    p=[r[1] for r in recommendations] / sum([r[1] for r in recommendations])
                )
                service.track_click(user_id, experiment_id, variant, clicked_activity)
                clicks_count += 1
                
                # Simulate booking (60% of clicks)
                if np.random.random() < 0.6:
                    service.track_booking(user_id, experiment_id, variant, clicked_activity, confirmed=True)
                    bookings_count += 1
        
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue
    
    print(f"Simulated {interactions_count} impressions, {clicks_count} clicks, {bookings_count} bookings")

def main():
    """Main function to demonstrate A/B testing."""
    print("="*60)
    print("A/B TESTING FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Set up models
    models_data = setup_models()
    
    # Create A/B testing service
    service = ABTestingService()
    
    # Create experiment
    experiment_id = "rec_strategy_comparison_001"
    experiment = service.framework.create_experiment(
        experiment_id=experiment_id,
        name="Recommendation Strategy Comparison",
        description="Compare Two Tower vs Hybrid vs Sequence-based strategies",
        variants=['control', 'variant_a', 'variant_b'],
        traffic_split={'control': 0.33, 'variant_a': 0.33, 'variant_b': 0.34},
        metrics=['click_through_rate', 'conversion_rate', 'engagement_score'],
        duration_days=30
    )
    
    print(f"\nCreated experiment: {experiment.name}")
    print(f"Variants: {experiment.variants}")
    
    # Register strategies for each variant
    service.register_experiment_strategies(experiment_id, {
        'control': {
            'strategy': 'two_tower',
            'trainer': models_data['two_tower_trainer'],
            'activity_to_idx': models_data['activity_to_idx']
        },
        'variant_a': {
            'strategy': 'hybrid',
            'two_tower_trainer': models_data['two_tower_trainer'],
            'sequence_trainer': models_data['sequence_trainer'],
            'activity_to_idx': models_data['activity_to_idx'],
            'weights': {'two_tower': 0.6, 'sequence': 0.4}
        },
        'variant_b': {
            'strategy': 'sequence_based',
            'trainer': models_data['sequence_trainer']
        }
    })
    
    print("\nRegistered strategies:")
    print("  - control: Two Tower model")
    print("  - variant_a: Hybrid model (Two Tower + Sequence)")
    print("  - variant_b: Sequence-based model")
    
    # Simulate user interactions
    simulate_user_interactions(service, experiment_id, models_data, n_users=200)
    
    # Get experiment results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    metrics = service.get_experiment_metrics(experiment_id)
    for variant, variant_metrics in metrics.items():
        print(f"\n{variant.upper()}:")
        print(f"  Impressions: {variant_metrics.get('impressions', 0)}")
        print(f"  Clicks: {variant_metrics.get('clicks', 0)}")
        print(f"  Bookings: {variant_metrics.get('bookings', 0)}")
        print(f"  Click-Through Rate: {variant_metrics.get('click_through_rate', 0):.4f}")
        print(f"  Conversion Rate: {variant_metrics.get('conversion_rate', 0):.4f}")
        print(f"  Booking Rate: {variant_metrics.get('booking_rate', 0):.4f}")
        print(f"  Engagement Score: {variant_metrics.get('engagement_score', 0):.4f}")
    
    # Compare variants
    comparison = service.get_experiment_results(experiment_id)
    print("\n" + "="*60)
    print("VARIANT COMPARISON")
    print("="*60)
    
    for variant, comp_data in comparison['comparison'].items():
        print(f"\n{variant} vs {comp_data['control']}:")
        print(f"  CTR Improvement: {comp_data['ctr_improvement']:.2f}%")
        print(f"  Conversion Improvement: {comp_data['conversion_improvement']:.2f}%")
        print(f"  Engagement Improvement: {comp_data['engagement_improvement']:.2f}%")
    
    print("\n" + "="*60)
    print("A/B TESTING DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()

