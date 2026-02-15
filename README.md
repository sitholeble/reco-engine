# Recommendation Engine

A comprehensive recommendation system for fitness activities with multiple modeling approaches.

## Features

1. **Collaborative Filtering** - Amazon-style "people who did X also did Y"
2. **Two Towers Model** - Deep learning model with separate towers for users and activities
3. **LSTM Sequence Model** - Predicts next activity based on activity sequences
4. **Hybrid Model** - Combines multiple approaches for better recommendations
5. **Feature Engineering** - Utilities for creating and transforming features
6. **A/B Testing Framework** - Evaluate different recommendation strategies and refine models based on user interactions

## Installation

### Option 1: Using Virtual Environment (Recommended)

On macOS with Python 3.13+, you need to use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or use the setup script:
```bash
./setup_venv.sh
```

**To activate the virtual environment in future sessions:**
```bash
source venv/bin/activate
```

**To deactivate:**
```bash
deactivate
```

### Option 2: System-wide Installation (Not Recommended)

If you must install system-wide (not recommended on macOS):
```bash
pip3 install -r requirements.txt --break-system-packages
```

## Testing

### Quick Start (3 Steps)

#### Step 1: Quick Validation (No Installation Needed)
```bash
python3 test_quick.py
```
Validates code structure and syntax without requiring dependencies.

#### Step 2: Setup Virtual Environment and Install Dependencies

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

#### Step 3: Run Full Tests
```bash
python3 test_system.py
```
Tests all functionality including new data models

---

### Testing Options

#### Option A: Automated Test Runner (Recommended)
```bash
python3 run_tests.py
```
Runs everything automatically:
- Checks dependencies
- Validates code structure
- Tests functionality
- Generates test data
- Optionally runs demo

#### Option B: Manual Step-by-Step

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

#### Option C: Complete Setup and Test
```bash
./setup_and_test.sh
```
Automated script that sets up virtual environment, installs dependencies, runs all tests, generates data, and optionally runs the demo.

---

### What Gets Tested

#### Data Generation
- Creates two user types (new users vs long-standing members)
- New users have skeletal data (age, weight, optional height)
- New users have missing metadata
- Long-standing members have robust metadata
- Activity distribution matches user types

#### Feature Engineering
- Handles missing data gracefully
- Imputes missing values
- Works with users who have no activities
- Creates features for both user types

#### Models
- Two Towers model works with minimal data (age, weight)
- Two Towers model works with full data (age, weight, height)
- Recommendations work for both user types
- Handles missing values automatically

---

### 🔍 Verify Test Results

#### Quick Test Should Show:
```
All files have valid Python syntax
All expected classes found
All expected functions found
CODE STRUCTURE VALIDATED
```

#### Full Test Should Show:
```
All dependencies installed
Code structure OK
Code syntax OK
Basic functionality works
ALL TESTS PASSED - System ready to use!
```

---

### Troubleshooting

**"ModuleNotFoundError"**
→ Make sure virtual environment is activated and dependencies are installed:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**"externally-managed-environment" error**
→ Use a virtual environment (see Step 2 above)

**"FileNotFoundError: data/user_profiles.csv"**
→ Generate data first:
  ```bash
  python3 data_generator.py
  ```

**"ValueError: Input contains NaN"**
→ Make sure to use `handle_missing=True` in feature engineering (already done in tests)

**"Empty activity sequences"**
→ This is expected for new users. The system handles this:
  ```python
  # Activity features will have zeros for users with no activities
  activity_features = fe.create_activity_features(activity_sequences, user_profiles)
  ```

---

### Manual Testing Examples

#### Test Data Generation
```python
import pandas as pd

# Check user types
profiles = pd.read_csv('data/user_profiles.csv')
print(f"New users: {len(profiles[profiles['membership_type'] == 'new_user'])}")
print(f"Long-standing: {len(profiles[profiles['membership_type'] == 'long_standing'])}")

# Check new users have missing data
new_users = profiles[profiles['membership_type'] == 'new_user']
print(f"New users with missing metadata: {new_users['membership_duration_days'].isna().sum()}")
```

#### Test Feature Engineering
```python
from feature_engineering import FeatureEngineer
import pandas as pd

# Load data
user_profiles = pd.read_csv('data/user_profiles.csv')
activity_sequences = pd.read_csv('data/activity_sequences.csv')

# Create feature engineer
fe = FeatureEngineer()

# Test with missing data
user_features = fe.create_user_features(user_profiles, handle_missing=True)
print(f"Created {len(user_features.columns)} features")
print(f"Handled missing values: {user_features.isna().sum().sum() == 0}")

# Test activity features (handles users with no activities)
activity_features = fe.create_activity_features(activity_sequences, user_profiles)
print(f"Activity features for all users: {len(activity_features) == len(user_profiles)}")
```

#### Test Models with Both User Types
```python
from models.two_towers import TwoTowerModel, TwoTowerTrainer
import pandas as pd

# Load data
user_profiles = pd.read_csv('data/user_profiles.csv')
interaction_matrix = pd.read_csv('data/interaction_matrix.csv')

# Prepare activity mapping
activities = sorted(interaction_matrix['activity'].unique())
activity_to_idx = {act: idx for idx, act in enumerate(activities)}

# Create model
model = TwoTowerModel(user_feature_dim=3, num_activities=len(activities), embedding_dim=64)
trainer = TwoTowerTrainer(model)

# Prepare data (handles missing values)
user_features, activity_indices, labels = trainer.prepare_data(
    user_profiles, 
    interaction_matrix, 
    activity_to_idx,
    use_feature_engineering=True
)

# Test recommendation for new user (minimal data)
new_user = user_profiles[user_profiles['membership_type'] == 'new_user'].iloc[0]
recs = trainer.recommend(
    [new_user['age'], new_user['weight_kg']],  # Only age and weight
    activity_to_idx,
    top_k=5
)
print(f"Recommendations for new user: {recs}")
```

## Quick Start


1. Setup virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Generate fake data:
```bash
python3 data_generator.py
```

3. Run the demo:
```bash
python3 demo.py
```

Or use the automated setup:
```bash
./setup_and_test.sh
```

## Data Structure

The system uses tabular CSV data with support for two user types:

### User Profiles (`user_profiles.csv`)

**All Users (Base Fields):**
- `user_id`: Unique user identifier
- `age`: User age
- `weight_kg`: User weight in kilograms
- `height_cm`: User height in centimeters (may be missing for some new users)
- `gender`: User gender (M, F, Other)
- `membership_type`: Either `'new_user'` or `'long_standing'`

**New Users (Skeletal Data):**
- Only base fields are populated
- Many fields are `None`/missing
- Typically have 0-3 activities

**Long-Standing Members (Robust Metadata):**
- `membership_duration_days`: Days since membership started
- `total_classes_attended`: Total number of classes attended
- `fitness_goal`: Primary fitness goal (weight_loss, muscle_gain, endurance, etc.)
- `preferred_time_of_day`: Preferred workout time (morning, afternoon, evening, flexible)
- `equipment_preference`: Equipment preference (minimal, full_gym, cardio_focused, etc.)
- `favorite_activities`: Comma-separated list of favorite activities
- `injury_history`: Any injury history (none, knee, back, shoulder, ankle)
- `activity_level`: Experience level (beginner, intermediate, advanced)
- `avg_rating_given`: Average rating given to activities
- `preferred_duration_minutes`: Preferred workout duration
- `membership_tier`: Membership tier (basic, premium, elite)
- `has_trainer`: Whether user has a personal trainer
- `nutrition_tracking`: Whether user tracks nutrition

### Activity Sequences (`activity_sequences.csv`)
- `user_id`: User identifier
- `activity`: Activity name
- `date`: Activity date
- `duration_minutes`: Duration of activity
- `rating`: User rating (1-5)

### Interaction Matrix (`interaction_matrix.csv`)
- `user_id`: User identifier
- `activity`: Activity name
- `count`: Number of times user did this activity
- `avg_rating`: Average rating for this user-activity pair

## Models

### 1. Collaborative Filtering

Simple item-based collaborative filtering using cosine similarity. Good for:
- Finding similar activities
- "People like you also do..." recommendations
- Popular activity discovery

### 2. Two Towers Model

Deep learning model with:
- **User Tower**: Encodes user features (age, weight, height) into embeddings
- **Activity Tower**: Encodes activities into embeddings
- **Similarity**: Uses dot product to find matching activities

Good for:
- Personalized recommendations based on user profile
- Handling cold-start problems with user features

### 3. LSTM Sequence Model

Recurrent neural network that predicts next activity in a sequence. Good for:
- "What should I do next?" predictions
- Sequential pattern learning
- Temporal recommendations

## Usage Examples

### Collaborative Filtering

```python
from models.collaborative_filtering import CollaborativeFiltering

cf = CollaborativeFiltering()
cf.fit(interaction_matrix)

# Find similar activities
similar = cf.recommend_similar_activities('spin_class', top_k=5)

# Recommend for user
recs = cf.recommend_for_user(user_id=1, top_k=5)
```

### Two Towers

```python
from models.two_towers import TwoTowerModel, TwoTowerTrainer

model = TwoTowerModel(user_feature_dim=3, num_activities=15, embedding_dim=64)
trainer = TwoTowerTrainer(model)

# Prepare data (handles missing values automatically)
user_features, activity_indices, labels = trainer.prepare_data(
    user_profiles, 
    interaction_matrix, 
    activity_to_idx,
    use_feature_engineering=True
)

# Train
trainer.train(user_features, activity_indices, labels)

# Recommend for new user (minimal data: age, weight)
recs = trainer.recommend([30, 70], activity_to_idx, top_k=5)

# Recommend for user with full data (age, weight, height)
recs = trainer.recommend([30, 70, 175], activity_to_idx, top_k=5)
```

### Sequence Model

```python
from models.sequence_model import ActivityLSTM, SequenceModelTrainer

model = ActivityLSTM(vocab_size=15, embedding_dim=64, hidden_dim=128)
trainer = SequenceModelTrainer(model)

# Train
trainer.train(X, y)

# Predict next activity
predictions = trainer.predict_next(['cycling', 'spin_class', 'weight_training'], top_k=5)
```

## Feature Engineering

The `FeatureEngineer` class provides utilities for:

- **Handling Missing Data**: Automatic imputation for new users with skeletal data
- **Creating User Features**: BMI, age groups, BMI categories, membership metadata
- **Creating Activity Features**: Frequency, diversity, trends (handles users with no activities)
- **Normalization**: Standard (z-score) or min-max scaling
- **Categorical Encoding**: Label encoding with handling for unseen categories
- **PCA**: Dimensionality reduction
- **Model-Ready Features**: `prepare_model_features()` method that handles both user types

### Handling New Users vs Long-Standing Members

The feature engineering system automatically:
- Imputes missing values (e.g., missing height for new users)
- Creates default features for users with no activity history
- Uses minimal features (age, weight, height) for new users when needed
- Leverages rich metadata for long-standing members

Example:
```python
from feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Create features (handles missing data automatically)
user_features, feature_cols = fe.prepare_model_features(
    user_profiles, 
    activity_sequences
)

# Get minimal features for new users
minimal_features = fe.get_minimal_features(user_profiles)
```

## A/B Testing Framework

Evaluate different recommendation strategies and refine models based on user interactions.

### Quick Start

```python
from ab_testing_service import ABTestingService

# Initialize service
service = ABTestingService()

# Create experiment
experiment = service.framework.create_experiment(
    experiment_id="exp_001",
    name="Two Tower vs Hybrid",
    description="Compare different recommendation strategies",
    variants=['control', 'variant_a'],
    traffic_split={'control': 0.5, 'variant_a': 0.5},
    metrics=['click_through_rate', 'conversion_rate', 'engagement_score'],
    duration_days=30
)

# Register strategies for each variant
service.register_experiment_strategies("exp_001", {
    'control': {
        'strategy': 'two_tower',
        'trainer': two_tower_trainer,
        'activity_to_idx': activity_to_idx
    },
    'variant_a': {
        'strategy': 'hybrid',
        'two_tower_trainer': two_tower_trainer,
        'sequence_trainer': sequence_trainer,
        'activity_to_idx': activity_to_idx
    }
})

# Get recommendations (automatically tracks impressions)
recommendations, variant = service.get_recommendations(
    user_id="user_123",
    experiment_id="exp_001",
    user_features={'age': 30, 'weight_kg': 70, 'height_cm': 170},
    activity_history=['cycling', 'running'],
    top_k=5
)

# Track user interactions
service.track_click("user_123", "exp_001", variant, "spin_class")
service.track_booking("user_123", "exp_001", variant, "spin_class", confirmed=True)

# Analyze results
results = service.get_experiment_results("exp_001")
```

### Available Strategies

- **two_tower**: Two Tower model
- **sequence_based**: LSTM/Sequence model  
- **hybrid**: Combines multiple models
- **collaborative_filtering**: Collaborative filtering
- **popularity_based**: Baseline using popular activities
- **diversity_focused**: Emphasizes diversity in recommendations

### Running Tests

```bash
# Run A/B testing demonstration
python test_ab_testing.py

# Analyze experiment results
python analyze_ab_results.py exp_001 --plot --export results.csv
```

### Key Features

- **Consistent User Assignment**: Same user always gets same variant (using consistent hashing)
- **Automatic Tracking**: Impressions tracked automatically when recommendations are generated
- **Multiple Metrics**: CTR, conversion rate, booking rate, engagement score
- **Easy Comparison**: Built-in comparison between variants with improvement percentages
- **Persistent Storage**: All data saved to `data/ab_testing/` directory

### Workflow

1. **Create Experiment**: Define variants and traffic split
2. **Register Strategies**: Map each variant to a recommendation strategy
3. **Serve Recommendations**: Use `get_recommendations()` to serve recommendations (tracks impressions)
4. **Track Interactions**: Call `track_click()` and `track_booking()` when users interact
5. **Analyze Results**: Use `get_experiment_results()` to compare variant performance

## Off-the-Shelf Alternatives

For production, consider:

- **PyTorch Lightning** - For easier model training
- **TensorFlow Recommenders** - Google's recommendation library
- **Surprise** - Scikit-learn compatible recommendation library
- **Implicit** - Fast collaborative filtering
- **LightFM** - Hybrid recommendation library


