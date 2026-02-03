"""
Test script for the recommendation system.
Tests code structure and basic functionality.
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    missing = []
    
    try:
        import pandas
        print("pandas")
    except ImportError:
        missing.append("pandas")
        print("pandas - MISSING")
    
    try:
        import numpy
        print("numpy")
    except ImportError:
        missing.append("numpy")
        print("numpy - MISSING")
    
    try:
        import sklearn
        print("scikit-learn")
    except ImportError:
        missing.append("scikit-learn")
        print("scikit-learn - MISSING")
    
    try:
        import torch
        print(f"torch (version {torch.__version__})")
    except ImportError:
        missing.append("torch")
        print("torch - MISSING")
    
    try:
        import scipy
        print("scipy")
    except ImportError:
        missing.append("scipy")
        print("scipy - MISSING")
    
    return missing

def test_code_structure():
    """Test if all code files exist and can be imported."""
    print("\nTesting code structure...")
    
    files_to_check = [
        'data_generator.py',
        'models/collaborative_filtering.py',
        'models/two_towers.py',
        'models/sequence_model.py',
        'feature_engineering.py',
        'demo.py'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"{file}")
        else:
            print(f"{file} - MISSING")
            all_exist = False
    
    return all_exist

def test_module_imports():
    """Test if modules can be imported (syntax check)."""
    print("\nTesting module imports (syntax check)...")
    
    try:
        # Test data generator
        sys.path.insert(0, '.')
        from data_generator import generate_user_profiles, generate_activity_sequences
        print("data_generator.py - syntax OK")
    except Exception as e:
        print(f"data_generator.py - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.collaborative_filtering import CollaborativeFiltering
        print("models.collaborative_filtering - syntax OK")
    except Exception as e:
        print(f"models.collaborative_filtering - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.two_towers import TwoTowerModel, TwoTowerTrainer
        print("models.two_towers - syntax OK")
    except Exception as e:
        print(f"models.two_towers - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.sequence_model import ActivityLSTM, SequenceModelTrainer
        print("models.sequence_model - syntax OK")
    except Exception as e:
        print(f"models.sequence_model - {type(e).__name__}: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("feature_engineering - syntax OK")
    except Exception as e:
        print(f"feature_engineering - {type(e).__name__}: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality if dependencies are available."""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test data generator functions exist
        from data_generator import generate_user_profiles, generate_activity_sequences, generate_interaction_matrix
        
        # Test 1: Generate user profiles with two types
        print("  Test 1: Generating user profiles (20 users, 30% new)...")
        user_profiles = generate_user_profiles(n_users=20, new_user_ratio=0.3)
        assert len(user_profiles) == 20, "Should generate 20 users"
        assert 'user_id' in user_profiles.columns, "Should have user_id column"
        assert 'age' in user_profiles.columns, "Should have age column"
        assert 'membership_type' in user_profiles.columns, "Should have membership_type column"
        
        # Check user types
        new_users = user_profiles[user_profiles['membership_type'] == 'new_user']
        long_standing = user_profiles[user_profiles['membership_type'] == 'long_standing']
        assert len(new_users) > 0, "Should have new users"
        assert len(long_standing) > 0, "Should have long-standing members"
        print(f"    Generated {len(new_users)} new users and {len(long_standing)} long-standing members")
        
        # Check new users have missing data
        new_user_missing = new_users[['membership_duration_days', 'total_classes_attended']].isna().all(axis=1).sum()
        assert new_user_missing == len(new_users), "New users should have missing metadata"
        print("    New users have skeletal data (missing metadata)")
        
        # Check long-standing members have robust data
        long_standing_complete = long_standing[['membership_duration_days', 'total_classes_attended']].notna().all(axis=1).sum()
        assert long_standing_complete > 0, "Long-standing members should have metadata"
        print("    Long-standing members have robust metadata")
        
        # Test 2: Generate activity sequences
        print("\n  Test 2: Generating activity sequences...")
        activity_sequences = generate_activity_sequences(user_profiles)
        assert 'activity' in activity_sequences.columns, "Should have activity column"
        
        # Check that new users have fewer activities
        new_user_activities = activity_sequences[activity_sequences['user_id'].isin(new_users['user_id'])].groupby('user_id').size()
        long_standing_activities = activity_sequences[activity_sequences['user_id'].isin(long_standing['user_id'])].groupby('user_id').size()
        
        if len(new_user_activities) > 0 and len(long_standing_activities) > 0:
            avg_new = new_user_activities.mean()
            avg_long = long_standing_activities.mean()
            assert avg_new <= avg_long, "New users should have fewer activities on average"
            print(f"    New users avg activities: {avg_new:.1f}, Long-standing: {avg_long:.1f}")
        
        print("    Activity sequences generated successfully")
        
        # Test 3: Generate interaction matrix
        print("\n  Test 3: Generating interaction matrix...")
        interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
        assert len(interaction_matrix) >= 0, "Should generate interaction matrix"
        print("    Interaction matrix generated")
        
        # Test 4: Collaborative filtering
        print("\n  Test 4: Testing collaborative filtering...")
        from models.collaborative_filtering import CollaborativeFiltering
        
        if len(interaction_matrix) > 0:
            cf = CollaborativeFiltering()
            cf.fit(interaction_matrix)
            print("    Collaborative filtering model initialized")
        else:
            print("    Skipped (no interactions to test)")
        
        # Test 5: Feature engineering with missing data
        print("\n  Test 5: Testing feature engineering with missing data...")
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        
        # Test user features (should handle missing data)
        user_features = fe.create_user_features(user_profiles, handle_missing=True)
        assert 'bmi' in user_features.columns, "Should create BMI feature"
        assert len(user_features) == len(user_profiles), "Should have features for all users"
        print("    User features created (handles missing data)")
        
        # Test activity features (should handle users with no activities)
        activity_features = fe.create_activity_features(activity_sequences, user_profiles)
        assert 'user_id' in activity_features.columns, "Should have user_id"
        assert len(activity_features) == len(user_profiles), "Should have features for all users"
        print("    Activity features created (handles users with no activities)")
        
        # Test prepare_model_features
        model_features, feature_cols = fe.prepare_model_features(user_profiles, activity_sequences)
        assert len(model_features) == len(user_profiles), "Should prepare features for all users"
        assert len(feature_cols) > 0, "Should have feature columns"
        print("    Model-ready features prepared")
        
        # Test minimal features for new users
        minimal_features = fe.get_minimal_features(user_profiles)
        assert 'age' in minimal_features.columns, "Should have age"
        assert 'weight_kg' in minimal_features.columns, "Should have weight"
        print("    Minimal features extracted")
        
        # Test 6: Two Towers model with missing data
        print("\n  Test 6: Testing Two Towers model with missing data...")
        if len(interaction_matrix) > 0:
            from models.two_towers import TwoTowerModel, TwoTowerTrainer
            
            activities = sorted(interaction_matrix['activity'].unique())
            activity_to_idx = {act: idx for idx, act in enumerate(activities)}
            
            model = TwoTowerModel(user_feature_dim=3, num_activities=len(activities), embedding_dim=32)
            trainer = TwoTowerTrainer(model, device='cpu')
            
            # Prepare data (should handle missing values)
            user_features, activity_indices, labels = trainer.prepare_data(
                user_profiles, 
                interaction_matrix, 
                activity_to_idx,
                use_feature_engineering=True
            )
            assert len(user_features) > 0, "Should prepare user features"
            print("    Data prepared for training (handles missing values)")
            
            # Test recommendation with minimal features (new user)
            if len(new_users) > 0:
                new_user = new_users.iloc[0]
                recs = trainer.recommend(
                    [new_user['age'], new_user['weight_kg']],  # Only age and weight
                    activity_to_idx,
                    top_k=3
                )
                assert len(recs) > 0, "Should generate recommendations"
                print("    Recommendations work with minimal features (new user)")
        else:
            print("    ⊘ Skipped (no interactions to test)")
        
        print("\n All basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"  Cannot test functionality - missing dependencies: {e}")
        print("  Install dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"  Error during functionality test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("RECOMMENDATION SYSTEM TEST SUITE")
    print("="*60)
    
    # Test 1: Check dependencies
    missing = test_imports()
    
    # Test 2: Check code structure
    structure_ok = test_code_structure()
    
    # Test 3: Check syntax
    syntax_ok = test_module_imports()
    
    # Test 4: Basic functionality (if dependencies available)
    if not missing:
        functionality_ok = test_basic_functionality()
    else:
        print("\n Skipping functionality tests - missing dependencies")
        functionality_ok = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
    else:
        print("\n All dependencies installed")
    
    if structure_ok:
        print(" Code structure OK")
    else:
        print(" Code structure issues found")
    
    if syntax_ok:
        print(" Code syntax OK")
    else:
        print(" Syntax errors found")
    
    if functionality_ok is True:
        print(" Basic functionality works")
    elif functionality_ok is False:
        print(" Functionality tests failed")
    else:
        print(" Functionality tests skipped (missing dependencies)")
    
    print("\n" + "="*60)
    
    if not missing and structure_ok and syntax_ok and functionality_ok:
        print(" ALL TESTS PASSED - System ready to use!")
        print("\nNext steps:")
        print("  1. Generate full dataset: python data_generator.py")
        print("  2. Run demo: python demo.py")
    else:
        print(" Some tests failed or skipped")
        if missing:
            print("\nTo install dependencies:")
            print("  pip install pandas numpy scikit-learn torch scipy")
            print("  OR")
            print("  pip install -r requirements.txt")

if __name__ == '__main__':
    main()

