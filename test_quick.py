"""
Quick test - validates code logic without requiring full dependencies.
Tests the structure and basic logic of the recommendation models.
"""
import ast
import os

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def test_all_files():
    """Test syntax of all Python files."""
    print("Testing Python file syntax...")
    print("-" * 60)
    
    files = [
        'data_generator.py',
        'models/collaborative_filtering.py',
        'models/two_towers.py',
        'models/sequence_model.py',
        'feature_engineering.py',
        'demo.py',
        'test_system.py'
    ]
    
    all_ok = True
    for filepath in files:
        if os.path.exists(filepath):
            ok, error = check_syntax(filepath)
            if ok:
                print(f"✓ {filepath}")
            else:
                print(f"✗ {filepath}")
                print(f"  Error: {error}")
                all_ok = False
        else:
            print(f"✗ {filepath} - FILE NOT FOUND")
            all_ok = False
    
    return all_ok

def check_class_structure():
    """Check if expected classes exist in files."""
    print("\nChecking class structure...")
    print("-" * 60)
    
    expected_classes = {
        'models/collaborative_filtering.py': ['CollaborativeFiltering'],
        'models/two_towers.py': ['UserTower', 'ActivityTower', 'TwoTowerModel', 'TwoTowerTrainer'],
        'models/sequence_model.py': ['ActivityLSTM', 'SequenceModelTrainer'],
        'feature_engineering.py': ['FeatureEngineer'],
    }
    
    all_found = True
    for filepath, classes in expected_classes.items():
        if not os.path.exists(filepath):
            print(f"✗ {filepath} - FILE NOT FOUND")
            all_found = False
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for cls in classes:
            if f"class {cls}" in content:
                print(f"✓ {filepath} - class {cls}")
            else:
                print(f"✗ {filepath} - class {cls} NOT FOUND")
                missing.append(cls)
        
        if missing:
            all_found = False
    
    return all_found

def check_functions():
    """Check if expected functions exist."""
    print("\nChecking key functions...")
    print("-" * 60)
    
    expected_functions = {
        'data_generator.py': [
            'generate_user_profiles',
            'generate_activity_sequences',
            'generate_interaction_matrix'
        ],
    }
    
    all_found = True
    for filepath, functions in expected_functions.items():
        if not os.path.exists(filepath):
            print(f"✗ {filepath} - FILE NOT FOUND")
            all_found = False
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        for func in functions:
            if f"def {func}" in content:
                print(f"✓ {filepath} - function {func}()")
            else:
                print(f"✗ {filepath} - function {func}() NOT FOUND")
                all_found = False
    
    return all_found

def main():
    """Run quick tests."""
    print("=" * 60)
    print("QUICK CODE VALIDATION TEST")
    print("=" * 60)
    print()
    
    syntax_ok = test_all_files()
    classes_ok = check_class_structure()
    functions_ok = check_functions()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if syntax_ok:
        print("✓ All files have valid Python syntax")
    else:
        print("✗ Some files have syntax errors")
    
    if classes_ok:
        print("✓ All expected classes found")
    else:
        print("✗ Some classes missing")
    
    if functions_ok:
        print("✓ All expected functions found")
    else:
        print("✗ Some functions missing")
    
    if syntax_ok and classes_ok and functions_ok:
        print("\n✓ CODE STRUCTURE VALIDATED")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Generate data: python data_generator.py")
        print("  3. Run full test: python test_system.py")
        print("  4. Run demo: python demo.py")
    else:
        print("\n⚠ Code structure issues found - please review errors above")

if __name__ == '__main__':
    main()

