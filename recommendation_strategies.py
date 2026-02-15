"""
Different recommendation strategies for A/B testing.
Each strategy uses different models or approaches to generate recommendations.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import pandas as pd

class RecommendationStrategy:
    """Base class for recommendation strategies."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
    
    def recommend(self, user_id: str, user_features: Dict, 
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            user_features: Dictionary of user features (age, weight, etc.)
            activity_history: List of activities user has done
            top_k: Number of recommendations to return
            
        Returns:
            List of tuples (activity, score)
        """
        raise NotImplementedError

class TwoTowerStrategy(RecommendationStrategy):
    """Strategy using Two Tower model."""
    
    def __init__(self, trainer, activity_to_idx):
        super().__init__("two_tower")
        self.trainer = trainer
        self.activity_to_idx = activity_to_idx
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate recommendations using Two Tower model."""
        # Prepare user features
        user_feat = [
            user_features.get('age', 30),
            user_features.get('weight_kg', 70),
            user_features.get('height_cm', 170)
        ]
        
        # Get recommendations
        recommendations = self.trainer.recommend(
            user_feat,
            self.activity_to_idx,
            top_k=top_k,
            exclude_activities=set(activity_history)
        )
        
        return recommendations

class SequenceBasedStrategy(RecommendationStrategy):
    """Strategy using Sequence/LSTM model."""
    
    def __init__(self, trainer):
        super().__init__("sequence_based")
        self.trainer = trainer
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate recommendations using Sequence model."""
        if len(activity_history) < 2:
            # Not enough history, return empty or fallback
            return []
        
        # Use recent activities as sequence
        recent_activities = activity_history[-5:] if len(activity_history) >= 5 else activity_history
        
        # Prepare user features
        user_feat = [
            user_features.get('age', 30),
            1.0 if user_features.get('gender') == 'M' else 0.0,
            0.25, 0.25, 0.25, 0.25  # Default class preferences
        ]
        
        # Get predictions
        predictions = self.trainer.predict_next(
            recent_activities,
            top_k=top_k,
            user_features=user_feat
        )
        
        return predictions

class HybridStrategy(RecommendationStrategy):
    """Strategy using Hybrid model (combines multiple approaches)."""
    
    def __init__(self, two_tower_trainer, sequence_trainer, activity_to_idx, weights=None):
        super().__init__("hybrid")
        self.two_tower_trainer = two_tower_trainer
        self.sequence_trainer = sequence_trainer
        self.activity_to_idx = activity_to_idx
        self.weights = weights or {'two_tower': 0.6, 'sequence': 0.4}
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate recommendations using Hybrid approach."""
        # Get recommendations from both models
        two_tower_recs = {}
        sequence_recs = {}
        
        # Two Tower recommendations
        user_feat = [
            user_features.get('age', 30),
            user_features.get('weight_kg', 70),
            user_features.get('height_cm', 170)
        ]
        tt_recs = self.two_tower_trainer.recommend(
            user_feat,
            self.activity_to_idx,
            top_k=top_k * 2,
            exclude_activities=set(activity_history)
        )
        for activity, score in tt_recs:
            two_tower_recs[activity] = score
        
        # Sequence recommendations
        if len(activity_history) >= 2:
            recent_activities = activity_history[-5:] if len(activity_history) >= 5 else activity_history
            user_feat_seq = [
                user_features.get('age', 30),
                1.0 if user_features.get('gender') == 'M' else 0.0,
                0.25, 0.25, 0.25, 0.25
            ]
            seq_preds = self.sequence_trainer.predict_next(
                recent_activities,
                top_k=top_k * 2,
                user_features=user_feat_seq
            )
            for activity, prob in seq_preds:
                sequence_recs[activity] = prob
        
        # Combine scores
        all_activities = set(two_tower_recs.keys()) | set(sequence_recs.keys())
        combined_scores = {}
        
        for activity in all_activities:
            tt_score = two_tower_recs.get(activity, 0.0)
            seq_score = sequence_recs.get(activity, 0.0)
            
            # Normalize scores to [0, 1] range
            combined_score = (
                self.weights['two_tower'] * tt_score +
                self.weights['sequence'] * seq_score
            )
            combined_scores[activity] = combined_score
        
        # Sort and return top k
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_k]

class CollaborativeFilteringStrategy(RecommendationStrategy):
    """Strategy using Collaborative Filtering."""
    
    def __init__(self, cf_model):
        super().__init__("collaborative_filtering")
        self.cf_model = cf_model
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate recommendations using Collaborative Filtering."""
        recommendations = self.cf_model.recommend_for_user(user_id, top_k=top_k)
        return recommendations

class PopularityBasedStrategy(RecommendationStrategy):
    """Strategy using popularity-based recommendations (baseline)."""
    
    def __init__(self, popular_activities: List[Tuple[str, float]]):
        super().__init__("popularity_based")
        self.popular_activities = popular_activities
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate recommendations based on popularity."""
        # Filter out activities user has already done
        filtered = [
            (activity, score) for activity, score in self.popular_activities
            if activity not in activity_history
        ]
        return filtered[:top_k]

class DiversityFocusedStrategy(RecommendationStrategy):
    """Strategy that focuses on diversity in recommendations."""
    
    def __init__(self, base_strategy: RecommendationStrategy, diversity_weight: float = 0.3):
        super().__init__("diversity_focused")
        self.base_strategy = base_strategy
        self.diversity_weight = diversity_weight
    
    def recommend(self, user_id: str, user_features: Dict,
                 activity_history: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Generate diverse recommendations."""
        # Get base recommendations
        base_recs = self.base_strategy.recommend(user_id, user_features, activity_history, top_k=top_k * 2)
        
        # Group by category and select diverse set
        from data_generator import ACTIVITY_TO_CLASS
        
        recs_by_category = {}
        for activity, score in base_recs:
            category = ACTIVITY_TO_CLASS.get(activity, 'cardio')
            if category not in recs_by_category:
                recs_by_category[category] = []
            recs_by_category[category].append((activity, score))
        
        # Select top from each category
        diverse_recs = []
        per_category = max(1, top_k // len(recs_by_category))
        
        for category, recs in recs_by_category.items():
            sorted_recs = sorted(recs, key=lambda x: x[1], reverse=True)
            diverse_recs.extend(sorted_recs[:per_category])
        
        # Sort by score and return top k
        diverse_recs.sort(key=lambda x: x[1], reverse=True)
        return diverse_recs[:top_k]

class StrategyFactory:
    """Factory for creating recommendation strategies."""
    
    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> RecommendationStrategy:
        """
        Create a recommendation strategy by name.
        
        Args:
            strategy_name: Name of the strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            RecommendationStrategy instance
        """
        if strategy_name == "two_tower":
            return TwoTowerStrategy(kwargs['trainer'], kwargs['activity_to_idx'])
        elif strategy_name == "sequence_based":
            return SequenceBasedStrategy(kwargs['trainer'])
        elif strategy_name == "hybrid":
            return HybridStrategy(
                kwargs['two_tower_trainer'],
                kwargs['sequence_trainer'],
                kwargs['activity_to_idx'],
                kwargs.get('weights')
            )
        elif strategy_name == "collaborative_filtering":
            return CollaborativeFilteringStrategy(kwargs['cf_model'])
        elif strategy_name == "popularity_based":
            return PopularityBasedStrategy(kwargs['popular_activities'])
        elif strategy_name == "diversity_focused":
            base = StrategyFactory.create_strategy(kwargs['base_strategy'], **kwargs)
            return DiversityFocusedStrategy(base, kwargs.get('diversity_weight', 0.3))
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

