"""
Quick validation script to check code structure and imports.
"""
import sys
import os

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    print("-" * 60)
    
    modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
    ]
    
    missing = []
    for module_name, alias in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} - MISSING")
            missing.append(module_name)
    
    return missing

def check_local_imports():
    """Check if local modules can be imported."""
    print("\nChecking local module imports...")
    print("-" * 60)
    
    local_modules = [
        'data_generator',
        'models.two_towers',
        'models.sequence_model',
        'models.hybrid_model',
        'evaluate_models',
        'feature_engineering',
    ]
    
    errors = []
    for module in local_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} - ERROR: {e}")
            errors.append((module, str(e)))
    
    return errors

def check_data_generator():
    """Check data generator functions."""
    print("\nChecking data generator functions...")
    print("-" * 60)
    
    try:
        from data_generator import (
            ACTIVITY_CLASSES, ACTIVITY_TO_CLASS, get_cardio_activities,
            generate_user_profiles, generate_activity_sequences, generate_interaction_matrix
        )
        
        print("✓ ACTIVITY_CLASSES defined")
        print(f"  Classes: {list(ACTIVITY_CLASSES.keys())}")
        
        print("✓ ACTIVITY_TO_CLASS defined")
        print(f"  Mapped {len(ACTIVITY_TO_CLASS)} activities")
        
        print("✓ get_cardio_activities function")
        cardio = get_cardio_activities()
        print(f"  Cardio activities: {len(cardio)} activities")
        
        print("✓ generate_user_profiles function")
        print("✓ generate_activity_sequences function")
        print("✓ generate_interaction_matrix function")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_models():
    """Check model classes."""
    print("\nChecking model classes...")
    print("-" * 60)
    
    try:
        from models.two_towers import TwoTowerModel, TwoTowerTrainer, UserTower, ActivityTower
        print("✓ TwoTowerModel")
        print("✓ TwoTowerTrainer")
        print("✓ UserTower")
        print("✓ ActivityTower")
    except Exception as e:
        print(f"✗ Two Towers: {e}")
    
    try:
        from models.sequence_model import ActivityLSTM, SequenceModelTrainer
        print("✓ ActivityLSTM")
        print("✓ SequenceModelTrainer")
    except Exception as e:
        print(f"✗ Sequence Model: {e}")
    
    try:
        from models.hybrid_model import HybridModel, HybridTrainer
        print("✓ HybridModel")
        print("✓ HybridTrainer")
    except Exception as e:
        print(f"✗ Hybrid Model: {e}")

def check_evaluation():
    """Check evaluation functions."""
    print("\nChecking evaluation functions...")
    print("-" * 60)
    
    try:
        from evaluate_models import (
            evaluate_model_predictions, print_evaluation_results, compare_models,
            evaluate_binary_classification, evaluate_cardio_precision
        )
        print("✓ evaluate_model_predictions")
        print("✓ print_evaluation_results")
        print("✓ compare_models")
        print("✓ evaluate_binary_classification")
        print("✓ evaluate_cardio_precision")
    except Exception as e:
        print(f"✗ Evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all checks."""
    print("=" * 60)
    print("CODE VALIDATION")
    print("=" * 60)
    
    # Check external dependencies
    missing = check_imports()
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        print("\nContinuing with local module checks...")
    else:
        print("\n✓ All dependencies available")
    
    # Check local imports
    errors = check_local_imports()
    
    if errors:
        print(f"\n⚠ {len(errors)} local import errors found")
    else:
        print("\n✓ All local modules import successfully")
    
    # Check specific functionality
    if not missing:  # Only check if dependencies are available
        check_data_generator()
        check_models()
        check_evaluation()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    if missing:
        print("\nTo run the full test:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run: python test_all_models.py")
    elif errors:
        print("\n⚠ Some import errors found. Check the errors above.")
    else:
        print("\n✓ Code structure validated!")
        print("Run: python test_all_models.py")

if __name__ == '__main__':
    main()

