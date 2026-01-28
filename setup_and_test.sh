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

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."
echo "This may take a few minutes..."
python3 -m pip install --user pandas numpy scikit-learn torch scipy || {
    echo "⚠ pip install failed, trying alternative method..."
    pip3 install --user pandas numpy scikit-learn torch scipy || {
        echo "⚠ Installation failed. Please install manually:"
        echo "   pip install pandas numpy scikit-learn torch scipy"
        echo "   OR"
        echo "   pip install -r requirements.txt"
        exit 1
    }
}
echo "✓ Dependencies installed"
echo ""

# Step 3: Run quick validation
echo "Step 3: Running quick code validation..."
python3 test_quick.py
echo ""

# Step 4: Run full test suite
echo "Step 4: Running full test suite..."
python3 test_system.py
echo ""

# Step 5: Generate data
echo "Step 5: Generating test data..."
if [ ! -d "data" ]; then
    mkdir -p data
fi
python3 data_generator.py
echo ""

# Step 6: Run demo (optional, can be skipped if too slow)
echo "Step 6: Running demo (this may take a few minutes)..."
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

