"""
Service that integrates A/B testing with recommendation generation.
Handles user assignment, recommendation generation, and interaction tracking.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ab_testing import ABTestingFramework
from recommendation_strategies import RecommendationStrategy, StrategyFactory

class ABTestingService:
    """
    Service that combines A/B testing framework with recommendation strategies.
    """
    
    def __init__(self, data_dir='data/ab_testing'):
        self.framework = ABTestingFramework(data_dir)
        self.strategies: Dict[str, Dict[str, RecommendationStrategy]] = {}  # {experiment_id: {variant: strategy}}
    
    def register_experiment_strategies(self, experiment_id: str, 
                                      variant_strategies: Dict[str, Dict]):
        """
        Register recommendation strategies for each variant in an experiment.
        
        Args:
            experiment_id: Experiment identifier
            variant_strategies: Dict mapping variant name to strategy config
                               e.g., {'control': {'strategy': 'two_tower', ...},
                                      'variant_a': {'strategy': 'hybrid', ...}}
        """
        if experiment_id not in self.strategies:
            self.strategies[experiment_id] = {}
        
        for variant, config in variant_strategies.items():
            strategy_name = config['strategy']
            strategy = StrategyFactory.create_strategy(strategy_name, **config)
            self.strategies[experiment_id][variant] = strategy
    
    def get_recommendations(self, user_id: str, experiment_id: str,
                           user_features: Dict, activity_history: List[str],
                           top_k: int = 5) -> Tuple[List[Tuple[str, float]], str]:
        """
        Get recommendations for a user based on their assigned variant.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            user_features: User feature dictionary
            activity_history: List of activities user has done
            top_k: Number of recommendations
            
        Returns:
            Tuple of (recommendations, variant_name)
        """
        # Assign user to variant
        variant = self.framework.assign_user_to_variant(user_id, experiment_id)
        
        # Get strategy for this variant
        if experiment_id not in self.strategies or variant not in self.strategies[experiment_id]:
            raise ValueError(f"No strategy registered for variant {variant} in experiment {experiment_id}")
        
        strategy = self.strategies[experiment_id][variant]
        
        # Generate recommendations
        recommendations = strategy.recommend(
            user_id, user_features, activity_history, top_k
        )
        
        # Track impression
        recommended_activities = [activity for activity, _ in recommendations]
        self.framework.track_interaction(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            interaction_type='impression',
            recommended_activities=recommended_activities
        )
        
        return recommendations, variant
    
    def track_click(self, user_id: str, experiment_id: str, variant: str,
                   clicked_activity: str):
        """Track when a user clicks on a recommended activity."""
        # Get the recommendations that were shown
        interactions = [
            i for i in self.framework.interactions
            if i.user_id == user_id and 
            i.experiment_id == experiment_id and
            i.variant == variant and
            i.interaction_type == 'impression'
        ]
        
        if interactions:
            last_impression = interactions[-1]
            self.framework.track_interaction(
                user_id=user_id,
                experiment_id=experiment_id,
                variant=variant,
                interaction_type='click',
                recommended_activities=last_impression.recommended_activities,
                clicked_activity=clicked_activity
            )
    
    def track_booking(self, user_id: str, experiment_id: str, variant: str,
                     activity: str, confirmed: bool = True):
        """Track when a user books a recommended activity."""
        self.framework.track_interaction(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            interaction_type='booking',
            recommended_activities=[activity],
            clicked_activity=activity,
            booking_confirmed=confirmed
        )
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get results and analysis for an experiment."""
        return self.framework.compare_variants(experiment_id)
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict:
        """Get metrics for each variant in an experiment."""
        return self.framework.get_experiment_metrics(experiment_id)

