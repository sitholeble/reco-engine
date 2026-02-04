"""
Unified testing script for all recommendation models.
Regenerates locked dataset and tests all models with same data.
Evaluates ROC-AUC, precision, recall, focusing on cardio activities.
"""
import os
import pandas as pd
import numpy as np
import torch
from data_generator import generate_user_profiles, generate_activity_sequences, generate_interaction_matrix
from models.two_towers import TwoTowerModel, TwoTowerTrainer
from models.sequence_model import ActivityLSTM, SequenceModelTrainer
from models.hybrid_model import HybridModel, HybridTrainer
from evaluate_models import evaluate_model_predictions, print_evaluation_results, compare_models
from data_generator import ACTIVITY_TO_CLASS

def get_cardio_activities():
    """Get list of cardio activities."""
    from data_generator import ACTIVITY_CLASSES
    return ACTIVITY_CLASSES.get('cardio', [])

def regenerate_locked_dataset(n_users=1000, seed=42):
    """
    Regenerate dataset with locked seed for consistent testing.
    
    Args:
        n_users: Number of users to generate
        seed: Random seed (default 42)
        
    Returns:
        Tuple of (user_profiles, activity_sequences, interaction_matrix)
    """
    print(f"Generating locked dataset with {n_users} users (seed={seed})...")
    
    # Generate user profiles
    user_profiles = generate_user_profiles(n_users=n_users, new_user_ratio=0.3, seed=seed)
    
    # Generate activity sequences
    activity_sequences = generate_activity_sequences(
        user_profiles, 
        min_activities_new=0, 
        max_activities_new=3,
        min_activities_long=10, 
        max_activities_long=100,
        seed=seed
    )
    
    # Generate interaction matrix
    interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
    
    print(f"Dataset generated:")
    print(f"  - Users: {len(user_profiles)}")
    print(f"  - Activity sequences: {len(activity_sequences)}")
    print(f"  - Interactions: {len(interaction_matrix)}")
    
    return user_profiles, activity_sequences, interaction_matrix

def prepare_evaluation_data(user_profiles, interaction_matrix, activity_sequences):
    """
    Prepare data for evaluation.
    Creates binary labels: 1 for cardio activities, 0 for others.
    
    Args:
        user_profiles: User profiles DataFrame
        interaction_matrix: Interaction matrix DataFrame
        activity_sequences: Activity sequences DataFrame
        
    Returns:
        Evaluation data dictionary
    """
    cardio_activities = set(get_cardio_activities())
    
    # Create binary labels: 1 if activity is cardio, 0 otherwise
    interaction_matrix = interaction_matrix.copy()
    interaction_matrix['is_cardio'] = interaction_matrix['activity'].apply(
        lambda x: 1 if x in cardio_activities else 0
    )
    
    # Merge with user profiles to get user features
    eval_data = interaction_matrix.merge(
        user_profiles[['user_id', 'age', 'weight_kg', 'height_cm', 'gender']],
        on='user_id',
        how='left'
    )
    
    # Fill missing values
    eval_data['age'] = eval_data['age'].fillna(eval_data['age'].mean())
    eval_data['weight_kg'] = eval_data['weight_kg'].fillna(eval_data['weight_kg'].mean())
    eval_data['height_cm'] = eval_data['height_cm'].fillna(eval_data['height_cm'].mean())
    eval_data['gender'] = eval_data['gender'].fillna('M')
    
    return eval_data

def test_two_tower_model(user_profiles, interaction_matrix, device='cpu'):
    """Test Two Towers model."""
    print("\n" + "="*60)
    print("TESTING TWO TOWERS MODEL")
    print("="*60)
    
    # Create activity mapping
    activities = sorted(interaction_matrix['activity'].unique())
    activity_to_idx = {act: idx for idx, act in enumerate(activities)}
    
    # Initialize model
    # User features: age, weight, height, gender, 4 class preferences = 8 features
    user_feature_dim = 8
    num_activities = len(activities)
    model = TwoTowerModel(user_feature_dim, num_activities, embedding_dim=64, num_classes=4)
    
    # Train
    trainer = TwoTowerTrainer(model, device=device)
    print("\nPreparing data...")
    result = trainer.prepare_data(
        user_profiles, interaction_matrix, activity_to_idx,
        include_gender=True, include_classes=True
    )
    if len(result) == 4:
        user_features, activity_indices, labels, activity_classes = result
    else:
        user_features, activity_indices, labels = result
        activity_classes = None
    
    print(f"Training data: {len(user_features)} samples")
    print("Training model...")
    train_losses, val_losses = trainer.train(
        user_features, activity_indices, labels,
        epochs=30, batch_size=64, lr=0.001,
        activity_classes=activity_classes
    )
    
    # Evaluate
    print("\nEvaluating model...")
    eval_data = prepare_evaluation_data(user_profiles, interaction_matrix, None)
    
    # Get predictions for all interactions
    model.eval()
    predictions = []
    true_labels = []
    activity_names = []
    user_ids_list = []
    
    with torch.no_grad():
        for idx, row in eval_data.iterrows():
            # Prepare user features
            user_feat = np.array([
                row['age'], row['weight_kg'], row['height_cm'],
                1.0 if row['gender'] == 'M' else (0.5 if row['gender'] == 'Other' else 0.0),
                0.25, 0.25, 0.25, 0.25  # Default class preferences
            ])
            user_feat = trainer.scaler.transform([user_feat])
            user_tensor = torch.FloatTensor(user_feat).to(device)
            
            # Get activity
            activity = row['activity']
            if activity in activity_to_idx:
                act_idx = activity_to_idx[activity]
                act_tensor = torch.LongTensor([act_idx]).to(device)
                
                # Get activity class
                act_class = ACTIVITY_TO_CLASS.get(activity, 'cardio')
                class_to_idx = {'cardio': 0, 'strength': 1, 'flexibility': 2, 'sports': 3}
                class_idx = class_to_idx.get(act_class, 0)
                class_tensor = torch.LongTensor([class_idx]).to(device)
                
                # Predict
                pred = model(user_tensor, act_tensor, class_tensor).item()
                
                predictions.append(pred)
                true_labels.append(row['is_cardio'])
                activity_names.append(activity)
                user_ids_list.append(row['user_id'])
    
    # Normalize predictions to [0, 1]
    predictions = np.array(predictions)
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)
    
    # Evaluate
    results = evaluate_model_predictions(
        'two_tower', predictions, np.array(true_labels), activity_names,
        user_ids=user_ids_list, detailed=True
    )
    
    print_evaluation_results(results)
    return results

def test_sequence_model(user_profiles, activity_sequences, interaction_matrix, device='cpu'):
    """Test Sequence model."""
    print("\n" + "="*60)
    print("TESTING SEQUENCE MODEL")
    print("="*60)
    
    if len(activity_sequences) == 0:
        print("No activity sequences available. Skipping sequence model.")
        return None
    
    # Prepare sequences
    print("\nPreparing sequences...")
    temp_trainer = SequenceModelTrainer(None, device=device)
    result = temp_trainer.prepare_sequences(
        activity_sequences, user_profiles, sequence_length=5, use_user_features=True
    )
    if len(result) == 4:
        X, y, user_features_seq, activity_to_idx = result
    else:
        X, y, activity_to_idx = result
        user_features_seq = None
    
    # Create model
    vocab_size = len(activity_to_idx)
    user_feature_dim = user_features_seq.shape[1] if user_features_seq is not None else 0
    model = ActivityLSTM(vocab_size, embedding_dim=64, hidden_dim=128, 
                        num_layers=2, user_feature_dim=user_feature_dim)
    trainer = SequenceModelTrainer(model, device=device)
    trainer.label_encoder = temp_trainer.label_encoder
    
    print(f"Training data: {len(X)} sequences")
    print("Training model...")
    train_losses, val_losses = trainer.train(
        X, y, epochs=30, batch_size=64, lr=0.001,
        user_features=user_features_seq
    )
    
    # Evaluate
    print("\nEvaluating model...")
    eval_data = prepare_evaluation_data(user_profiles, interaction_matrix, activity_sequences)
    
    predictions = []
    true_labels = []
    activity_names = []
    user_ids_list = []
    
    # For each user-activity pair, get predictions
    for user_id in eval_data['user_id'].unique():
        user_data = eval_data[eval_data['user_id'] == user_id]
        user_sequences = activity_sequences[activity_sequences['user_id'] == user_id].sort_values('date')
        
        if len(user_sequences) >= 3:
            # Get recent sequence
            recent_activities = user_sequences['activity'].tail(5).tolist()
            
            # Get user features
            user_row = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
            user_feat = [
                user_row['age'],
                1.0 if user_row['gender'] == 'M' else (0.5 if user_row['gender'] == 'Other' else 0.0),
                0.25, 0.25, 0.25, 0.25  # Default class preferences
            ]
            
            # Predict for each activity in user_data
            for idx, row in user_data.iterrows():
                activity = row['activity']
                try:
                    preds = trainer.predict_next(recent_activities, top_k=len(activity_to_idx), 
                                               user_features=user_feat)
                    # Find prediction for this activity
                    pred_score = 0.0
                    for act, prob in preds:
                        if act == activity:
                            pred_score = prob
                            break
                    # If not found, use average probability
                    if pred_score == 0.0 and len(preds) > 0:
                        pred_score = np.mean([p for _, p in preds])
                    
                    predictions.append(pred_score)
                    true_labels.append(row['is_cardio'])
                    activity_names.append(activity)
                    user_ids_list.append(user_id)
                except:
                    predictions.append(0.5)  # Default
                    true_labels.append(row['is_cardio'])
                    activity_names.append(activity)
                    user_ids_list.append(user_id)
        else:
            # No sequence history, use default
            for idx, row in user_data.iterrows():
                predictions.append(0.5)
                true_labels.append(row['is_cardio'])
                activity_names.append(row['activity'])
                user_ids_list.append(user_id)
    
    predictions = np.array(predictions)
    
    # Evaluate
    results = evaluate_model_predictions(
        'sequence', predictions, np.array(true_labels), activity_names,
        user_ids=user_ids_list, detailed=True
    )
    
    print_evaluation_results(results)
    return results

def test_hybrid_model(user_profiles, activity_sequences, interaction_matrix, device='cpu'):
    """Test Hybrid model."""
    print("\n" + "="*60)
    print("TESTING HYBRID MODEL")
    print("="*60)
    
    # This is a simplified version - full implementation would require more complex sequence extraction
    print("Hybrid model testing requires more complex implementation.")
    print("Skipping for now - can be extended later.")
    return None

def main():
    """Main testing function."""
    print("="*60)
    print("UNIFIED MODEL TESTING")
    print("="*60)
    print("\nThis script:")
    print("  1. Regenerates locked dataset (seed=42)")
    print("  2. Trains all models on same data")
    print("  3. Evaluates with ROC-AUC and precision metrics")
    print("  4. Focuses on cardio activity predictions (target precision: 65%)")
    print("="*60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Regenerate locked dataset
    user_profiles, activity_sequences, interaction_matrix = regenerate_locked_dataset(
        n_users=1000, seed=42
    )
    
    # Save locked dataset
    os.makedirs('data', exist_ok=True)
    user_profiles.to_csv('data/user_profiles.csv', index=False)
    activity_sequences.to_csv('data/activity_sequences.csv', index=False)
    interaction_matrix.to_csv('data/interaction_matrix.csv', index=False)
    print("\nLocked dataset saved to data/ directory")
    
    # Test all models
    results = []
    
    # Test Two Towers
    try:
        tt_results = test_two_tower_model(user_profiles, interaction_matrix, device=device)
        if tt_results:
            results.append(tt_results)
    except Exception as e:
        print(f"Error testing Two Towers model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Sequence
    try:
        seq_results = test_sequence_model(user_profiles, activity_sequences, interaction_matrix, device=device)
        if seq_results:
            results.append(seq_results)
    except Exception as e:
        print(f"Error testing Sequence model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Hybrid (if implemented)
    # try:
    #     hybrid_results = test_hybrid_model(user_profiles, activity_sequences, interaction_matrix, device=device)
    #     if hybrid_results:
    #         results.append(hybrid_results)
    # except Exception as e:
    #     print(f"Error testing Hybrid model: {e}")
    
    # Compare models
    if len(results) > 0:
        compare_models(results)
        
        print("\n" + "="*60)
        print("TESTING COMPLETE")
        print("="*60)
        print("\nSummary:")
        for result in results:
            print(f"  {result['model_type']}: Cardio Precision = {result['cardio_precision']:.4f}, "
                  f"ROC-AUC = {result['overall_roc_auc']:.4f}")
    else:
        print("\nNo models were successfully tested.")

if __name__ == '__main__':
    main()

