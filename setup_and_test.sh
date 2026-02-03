#!/bin/bash
# Setup and test script for recommendation engine

set -e  # Exit on error

echo "============================================================"
echo "RECOMMENDATION ENGINE - SETUP AND TEST"
echo "============================================================"
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python3 --version
echo ""

# Step 2: Setup virtual environment
echo "Step 2: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated"
echo ""

# Step 3: Install dependencies
echo "Step 3: Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "⚠ Installation failed. Trying individual packages..."
    pip install pandas numpy scikit-learn torch scipy || {
        echo "⚠ Installation failed. Please check your internet connection and try again."
        exit 1
    }
}
echo "Dependencies installed"
echo ""

# Step 4: Run quick validation
echo "Step 4: Running quick code validation..."
python3 test_quick.py
echo ""

# Step 5: Run full test suite
echo "Step 5: Running full test suite..."
python3 test_system.py
echo ""

# Step 6: Generate data
echo "Step 6: Generating test data..."
if [ ! -d "data" ]; then
    mkdir -p data
fi
python3 data_generator.py
echo ""

# Step 7: Run demo (optional, can be skipped if too slow)
echo "Step 7: Running demo (this may take a few minutes)..."
read -p "Run full demo? This will train models (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 demo.py
else
    echo "Skipping demo. Run 'python3 demo.py' manually when ready."
fi

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Note: Virtual environment is still active."
echo "To deactivate, run: deactivate"
echo "To reactivate in future sessions, run: source venv/bin/activate"
echo ""

