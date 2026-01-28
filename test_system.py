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
        print("✓ pandas")
    except ImportError:
        missing.append("pandas")
        print("✗ pandas - MISSING")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy - MISSING")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        missing.append("scikit-learn")
        print("✗ scikit-learn - MISSING")
    
    try:
        import torch
        print(f"✓ torch (version {torch.__version__})")
    except ImportError:
        missing.append("torch")
        print("✗ torch - MISSING")
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError:
        missing.append("scipy")
        print("✗ scipy - MISSING")
    
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
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_exist = False
    
    return all_exist

def test_module_imports():
    """Test if modules can be imported (syntax check)."""
    print("\nTesting module imports (syntax check)...")
    
    try:
        # Test data generator
        sys.path.insert(0, '.')
        from data_generator import generate_user_profiles, generate_activity_sequences
        print("✓ data_generator.py - syntax OK")
    except Exception as e:
        print(f"✗ data_generator.py - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.collaborative_filtering import CollaborativeFiltering
        print("✓ models.collaborative_filtering - syntax OK")
    except Exception as e:
        print(f"✗ models.collaborative_filtering - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.two_towers import TwoTowerModel, TwoTowerTrainer
        print("✓ models.two_towers - syntax OK")
    except Exception as e:
        print(f"✗ models.two_towers - {type(e).__name__}: {e}")
        return False
    
    try:
        from models.sequence_model import ActivityLSTM, SequenceModelTrainer
        print("✓ models.sequence_model - syntax OK")
    except Exception as e:
        print(f"✗ models.sequence_model - {type(e).__name__}: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✓ feature_engineering - syntax OK")
    except Exception as e:
        print(f"✗ feature_engineering - {type(e).__name__}: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality if dependencies are available."""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test data generator functions exist
        from data_generator import generate_user_profiles, generate_activity_sequences
        
        # Generate small test dataset
        print("  Generating test data (10 users)...")
        user_profiles = generate_user_profiles(n_users=10)
        assert len(user_profiles) == 10, "Should generate 10 users"
        assert 'user_id' in user_profiles.columns, "Should have user_id column"
        assert 'age' in user_profiles.columns, "Should have age column"
        print("  ✓ User profiles generated successfully")
        
        activity_sequences = generate_activity_sequences(n_users=10, min_activities=3, max_activities=10)
        assert len(activity_sequences) > 0, "Should generate activity sequences"
        assert 'activity' in activity_sequences.columns, "Should have activity column"
        print("  ✓ Activity sequences generated successfully")
        
        # Test collaborative filtering
        from models.collaborative_filtering import CollaborativeFiltering
        from data_generator import generate_interaction_matrix
        
        interaction_matrix = generate_interaction_matrix(user_profiles, activity_sequences)
        cf = CollaborativeFiltering()
        cf.fit(interaction_matrix)
        print("  ✓ Collaborative filtering model initialized")
        
        # Test feature engineering
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        user_features = fe.create_user_features(user_profiles)
        assert 'bmi' in user_features.columns, "Should create BMI feature"
        print("  ✓ Feature engineering works")
        
        print("\n✓ All basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Cannot test functionality - missing dependencies: {e}")
        print("  Install dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"  ✗ Error during functionality test: {type(e).__name__}: {e}")
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
        print("\n⚠ Skipping functionality tests - missing dependencies")
        functionality_ok = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies installed")
    
    if structure_ok:
        print("✓ Code structure OK")
    else:
        print("✗ Code structure issues found")
    
    if syntax_ok:
        print("✓ Code syntax OK")
    else:
        print("✗ Syntax errors found")
    
    if functionality_ok is True:
        print("✓ Basic functionality works")
    elif functionality_ok is False:
        print("✗ Functionality tests failed")
    else:
        print("⚠ Functionality tests skipped (missing dependencies)")
    
    print("\n" + "="*60)
    
    if not missing and structure_ok and syntax_ok and functionality_ok:
        print("✓ ALL TESTS PASSED - System ready to use!")
        print("\nNext steps:")
        print("  1. Generate full dataset: python data_generator.py")
        print("  2. Run demo: python demo.py")
    else:
        print("⚠ Some tests failed or skipped")
        if missing:
            print("\nTo install dependencies:")
            print("  pip install pandas numpy scikit-learn torch scipy")
            print("  OR")
            print("  pip install -r requirements.txt")

if __name__ == '__main__':
    main()

