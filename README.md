# Recommendation Engine

A comprehensive recommendation system for fitness activities with multiple modeling approaches.

## Features

1. **Collaborative Filtering** - Amazon-style "people who did X also did Y"
2. **Two Towers Model** - Deep learning model with separate towers for users and activities
3. **LSTM Sequence Model** - Predicts next activity based on activity sequences
4. **Feature Engineering** - Utilities for creating and transforming features

## Installation

```bash
pip install -r requirements.txt
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
Runs comprehensive tests including functionality checks.

See [TESTING.md](TESTING.md) for detailed testing guide.

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

The system uses tabular CSV data:

- `user_profiles.csv`: User demographics (age, weight, height, gender)
- `activity_sequences.csv`: User activity history (user_id, activity, date, duration, rating)
- `interaction_matrix.csv`: User-activity interaction matrix

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

# Train
trainer.train(user_features, activity_indices, labels)

# Recommend
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

- Creating user features (BMI, age groups, BMI categories)
- Creating activity features (frequency, diversity, trends)
- Normalization (standard, min-max)
- Categorical encoding
- PCA for dimensionality reduction

## Off-the-Shelf Alternatives

For production, consider:

- **PyTorch Lightning** - For easier model training
- **TensorFlow Recommenders** - Google's recommendation library
- **Surprise** - Scikit-learn compatible recommendation library
- **Implicit** - Fast collaborative filtering
- **LightFM** - Hybrid recommendation library

## Next Steps

1. Add more sophisticated features (time of day, location, weather)
2. Implement hybrid models combining multiple approaches
3. Add evaluation metrics (precision@k, recall@k, NDCG)
4. Deploy as a REST API
5. Add real-time inference capabilities

## License

MIT

