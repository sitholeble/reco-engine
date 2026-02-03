# Recommendation Engine

A comprehensive recommendation system for fitness activities with multiple modeling approaches.

## Features

1. **Collaborative Filtering** - Amazon-style "people who did X also did Y"
2. **Two Towers Model** - Deep learning model with separate towers for users and activities
3. **LSTM Sequence Model** - Predicts next activity based on activity sequences
4. **Feature Engineering** - Utilities for creating and transforming features

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

### Quick Validation (No Dependencies)
```bash
python3 test_quick.py
```
Validates code structure and syntax without requiring dependencies.

### Full Testing (Requires Dependencies)
```bash
python3 test_system.py
```
Runs comprehensive tests including:
- Data generation with two user types (new users vs long-standing members)
- Missing data handling
- Feature engineering with incomplete data
- Model compatibility with both user types

### Automated Test Runner
```bash
python3 run_tests.py
```
Runs all tests in sequence and optionally generates data and runs demo.

### Testing Guide
See [TESTING.md](TESTING.md) for detailed testing instructions, manual testing examples, and troubleshooting.

## Quick Start

1. Generate fake data:
```bash
python data_generator.py
```

2. Run the demo:
```bash
python demo.py
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

## Off-the-Shelf Alternatives

For production, consider:

- **PyTorch Lightning** - For easier model training
- **TensorFlow Recommenders** - Google's recommendation library
- **Surprise** - Scikit-learn compatible recommendation library
- **Implicit** - Fast collaborative filtering
- **LightFM** - Hybrid recommendation library


