# Testing Guide

## Quick Validation

I've created a validation script to check the code structure:

```bash
python3 validate_code.py
```

This checks:
- ✓ Code syntax (all files compile)
- ✓ Import structure
- ✓ Function definitions
- ✓ Class definitions

## Full Testing

To run the complete test suite with all models:

### 1. Install Dependencies

```bash
# Activate virtual environment (if using venv)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unified Test

```bash
python test_all_models.py
```

This will:
1. ✅ Generate locked dataset (1000 users, seed=42)
2. ✅ Train Two Towers model (with age, gender, classes)
3. ✅ Train Sequence model (previous + current → future)
4. ✅ Evaluate with ROC-AUC and precision metrics
5. ✅ Focus on cardio activities (target: 65% precision)
6. ✅ Compare all models side-by-side

## Expected Output

```
====================================================================
UNIFIED MODEL TESTING
====================================================================

Generating locked dataset with 1000 users (seed=42)...
Dataset generated:
  - Users: 1000
  - Activity sequences: [number]
  - Interactions: [number]

====================================================================
TESTING TWO TOWERS MODEL
====================================================================

Preparing data...
Training data: [number] samples
Training model...
Epoch 10/30 - Train Loss: [value], Val Loss: [value]
...

Evaluating model...

====================================================================
EVALUATION RESULTS: TWO_TOWER
====================================================================

Overall Metrics:
  ROC-AUC: [0.XX]
  Precision: [0.XX]
  Recall: [0.XX]
  F1-Score: [0.XX]

Cardio-Specific Metrics:
  Cardio Precision: [0.XX] (Target: 0.65)
  Cardio Recall: [0.XX]
  Cardio F1-Score: [0.XX]
  Cardio ROC-AUC: [0.XX]
  Cardio Samples: [number]
  Meets Target Precision: ✓ YES / ✗ NO
  Optimal Threshold: [0.XX]

====================================================================
MODEL COMPARISON
====================================================================

Model               ROC-AUC    Precision    Cardio Prec   Meets Target
----------------------------------------------------------------------
two_tower           [value]    [value]      [value]       ✓/✗
sequence            [value]    [value]      [value]       ✓/✗
```

## What Gets Tested

### Two Towers Model
- ✅ Enhanced with age, gender, activity classes
- ✅ User features: age, weight, height, gender, 4 class preferences
- ✅ Activity tower with class embeddings
- ✅ ROC-AUC evaluation
- ✅ Cardio precision evaluation

### Sequence Model
- ✅ Uses previous + current activities to predict future
- ✅ Incorporates user features (age, gender, classes)
- ✅ LSTM with user feature concatenation
- ✅ ROC-AUC evaluation
- ✅ Cardio precision evaluation

## Troubleshooting

### Missing Dependencies
If you see `ModuleNotFoundError`:
```bash
pip install pandas numpy scikit-learn torch scipy
```

### Virtual Environment Issues
If venv activation fails:
```bash
# Use venv python directly
venv/bin/python test_all_models.py
```

### Import Errors
Check that all files are in the correct locations:
- `data_generator.py` (root)
- `models/two_towers.py`
- `models/sequence_model.py`
- `models/hybrid_model.py`
- `evaluate_models.py`
- `test_all_models.py`

## Code Validation Results

The validation script confirms:
- ✅ All Python files have valid syntax
- ✅ All imports are correctly structured
- ✅ All functions and classes are defined
- ⚠️ Dependencies need to be installed to run full tests

## Next Steps

1. **Install dependencies** in your environment
2. **Run the test**: `python test_all_models.py`
3. **Review results**: Check ROC-AUC and cardio precision scores
4. **Improve models**: If precision < 65%, tune hyperparameters or features

