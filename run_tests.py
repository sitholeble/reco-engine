"""
Automated test runner - runs all tests in sequence.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_dependencies():
    """Check if dependencies are installed."""
    print("\nChecking dependencies...")
    missing = []
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError:
        missing.append("pandas")
        print("✗ pandas")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        missing.append("scikit-learn")
        print("✗ scikit-learn")
    
    try:
        import torch
        print(f"✓ torch (version {torch.__version__})")
    except ImportError:
        missing.append("torch")
        print("✗ torch")
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError:
        missing.append("scipy")
        print("✗ scipy")
    
    return missing

def main():
    """Run all tests."""
    print("="*60)
    print("RECOMMENDATION ENGINE - AUTOMATED TEST RUNNER")
    print("="*60)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
        print("  OR")
        print("  pip install pandas numpy scikit-learn torch scipy")
        print("\nContinuing with available tests...")
    
    results = {}
    
    # Step 1: Quick validation
    results['quick_test'] = run_command(
        "python3 test_quick.py",
        "Step 1: Quick Code Validation"
    )
    
    # Step 2: Full test suite (if dependencies available)
    if not missing:
        results['full_test'] = run_command(
            "python3 test_system.py",
            "Step 2: Full Test Suite"
        )
        
        # Step 3: Generate data
        if not os.path.exists('data/user_profiles.csv'):
            results['generate_data'] = run_command(
                "python3 data_generator.py",
                "Step 3: Generate Test Data"
            )
        else:
            print("\n" + "="*60)
            print("Step 3: Data already exists, skipping generation")
            print("="*60)
            results['generate_data'] = True
        
        # Step 4: Run demo (optional)
        if results.get('generate_data'):
            print("\n" + "="*60)
            print("Step 4: Run Demo")
            print("="*60)
            response = input("\nRun full demo? This will train models (may take a few minutes) [y/N]: ")
            if response.lower() == 'y':
                results['demo'] = run_command(
                    "python3 demo.py",
                    "Running Demo"
                )
            else:
                print("Skipping demo. Run 'python3 demo.py' manually when ready.")
                results['demo'] = None
    else:
        print("\n⚠ Skipping full tests - missing dependencies")
        results['full_test'] = None
        results['generate_data'] = None
        results['demo'] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            print(f"✓ {test_name}")
        elif result is False:
            print(f"✗ {test_name} - FAILED")
        elif result is None:
            print(f"⊘ {test_name} - SKIPPED")
    
    all_passed = all(r for r in results.values() if r is not None)
    
    if all_passed and not missing:
        print("\n✓ ALL TESTS PASSED!")
        print("\nSystem is ready to use!")
    elif missing:
        print(f"\n⚠ Some tests skipped due to missing dependencies")
        print("Install dependencies and run again for full test suite")
    else:
        print("\n⚠ Some tests failed - please review errors above")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

