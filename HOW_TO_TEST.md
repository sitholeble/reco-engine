# How to Test This App

## 🚀 Quick Start (3 Steps)

### Step 1: Quick Validation (No Installation Needed)
```bash
python3 test_quick.py
```
✅ Validates code structure and syntax

### Step 2: Setup Virtual Environment and Install Dependencies

**On macOS (Python 3.13+), you need a virtual environment:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Or use the setup script:**
```bash
./setup_venv.sh
```

**Note:** Always activate the virtual environment before running tests:
```bash
source venv/bin/activate
```

### Step 3: Run Full Tests
```bash
python3 test_system.py
```
✅ Tests all functionality including new data models

---

## 📋 Testing Options

### Option A: Automated Test Runner (Recommended)
```bash
python3 run_tests.py
```
Runs everything automatically:
- ✅ Checks dependencies
- ✅ Validates code structure
- ✅ Tests functionality
- ✅ Generates test data
- ✅ Optionally runs demo

### Option B: Manual Step-by-Step

1. **Quick syntax check:**
   ```bash
   python3 test_quick.py
   ```

2. **Generate test data:**
   ```bash
   python3 data_generator.py
   ```
   Creates:
   - `data/user_profiles.csv` (new users + long-standing members)
   - `data/activity_sequences.csv`
   - `data/interaction_matrix.csv`

3. **Run full test suite:**
   ```bash
   python3 test_system.py
   ```

4. **Run demo (optional):**
   ```bash
   python3 demo.py
   ```

---

## ✅ What Gets Tested

### Data Generation
- ✅ Creates two user types (new users vs long-standing members)
- ✅ New users have skeletal data (age, weight, optional height)
- ✅ New users have missing metadata
- ✅ Long-standing members have robust metadata
- ✅ Activity distribution matches user types

### Feature Engineering
- ✅ Handles missing data gracefully
- ✅ Imputes missing values
- ✅ Works with users who have no activities
- ✅ Creates features for both user types

### Models
- ✅ Two Towers model works with minimal data (age, weight)
- ✅ Two Towers model works with full data (age, weight, height)
- ✅ Recommendations work for both user types
- ✅ Handles missing values automatically

---

## 🔍 Verify Test Results

### Quick Test Should Show:
```
✓ All files have valid Python syntax
✓ All expected classes found
✓ All expected functions found
✓ CODE STRUCTURE VALIDATED
```

### Full Test Should Show:
```
✓ All dependencies installed
✓ Code structure OK
✓ Code syntax OK
✓ Basic functionality works
✓ ALL TESTS PASSED - System ready to use!
```

---

## 🐛 Troubleshooting

**"ModuleNotFoundError"**
→ Make sure virtual environment is activated and dependencies are installed:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**"externally-managed-environment" error**
→ Use a virtual environment (see Step 2 above)

**"FileNotFoundError: data/user_profiles.csv"**
→ Generate data: `python3 data_generator.py`

**"ValueError: Input contains NaN"**
→ Use `handle_missing=True` in feature engineering (already done in tests)

---

## 📚 More Details

For comprehensive testing guide, see [TESTING.md](TESTING.md)

