"""
A/B Testing Framework for Recommendation System
Evaluates different recommendation strategies and tracks user interactions.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from dataclasses import dataclass, asdict
import pickle

@dataclass
class Experiment:
    """Represents an A/B testing experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[str]  # e.g., ['control', 'variant_a', 'variant_b']
    start_date: datetime
    end_date: Optional[datetime]
    status: str  # 'active', 'completed', 'paused'
    traffic_split: Dict[str, float]  # e.g., {'control': 0.5, 'variant_a': 0.5}
    metrics: List[str]  # e.g., ['click_through_rate', 'conversion_rate', 'engagement_score']
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        if isinstance(data['start_date'], str):
            data['start_date'] = datetime.fromisoformat(data['start_date'])
        if data['end_date'] and isinstance(data['end_date'], str):
            data['end_date'] = datetime.fromisoformat(data['end_date'])
        return cls(**data)

@dataclass
class UserInteraction:
    """Represents a user interaction with recommendations."""
    user_id: str
    experiment_id: str
    variant: str
    timestamp: datetime
    interaction_type: str  # 'impression', 'click', 'booking', 'skip'
    recommended_activities: List[str]
    clicked_activity: Optional[str] = None
    booking_confirmed: bool = False
    session_id: Optional[str] = None
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'variant': self.variant,
            'timestamp': self.timestamp.isoformat(),
            'interaction_type': self.interaction_type,
            'recommended_activities': self.recommended_activities,
            'clicked_activity': self.clicked_activity,
            'booking_confirmed': self.booking_confirmed,
            'session_id': self.session_id
        }

class ABTestingFramework:
    """
    A/B Testing framework for recommendation strategies.
    Manages experiments, assigns users to variants, and tracks metrics.
    """
    
    def __init__(self, data_dir='data/ab_testing'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self.interactions: List[UserInteraction] = []
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # {user_id: {experiment_id: variant}}
        self._load_data()
    
    def _load_data(self):
        """Load existing experiments and interactions from disk."""
        # Load experiments
        experiments_file = os.path.join(self.data_dir, 'experiments.json')
        if os.path.exists(experiments_file):
            with open(experiments_file, 'r') as f:
                data = json.load(f)
                for exp_id, exp_data in data.items():
                    self.experiments[exp_id] = Experiment.from_dict(exp_data)
        
        # Load interactions
        interactions_file = os.path.join(self.data_dir, 'interactions.csv')
        if os.path.exists(interactions_file):
            df = pd.read_csv(interactions_file)
            for _, row in df.iterrows():
                interaction = UserInteraction(
                    user_id=row['user_id'],
                    experiment_id=row['experiment_id'],
                    variant=row['variant'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    interaction_type=row['interaction_type'],
                    recommended_activities=json.loads(row['recommended_activities']),
                    clicked_activity=row.get('clicked_activity'),
                    booking_confirmed=row.get('booking_confirmed', False),
                    session_id=row.get('session_id')
                )
                self.interactions.append(interaction)
        
        # Load user assignments
        assignments_file = os.path.join(self.data_dir, 'user_assignments.json')
        if os.path.exists(assignments_file):
            with open(assignments_file, 'r') as f:
                self.user_assignments = json.load(f)
    
    def _save_data(self):
        """Save experiments and interactions to disk."""
        # Save experiments
        experiments_file = os.path.join(self.data_dir, 'experiments.json')
        experiments_dict = {
            exp_id: exp.to_dict() for exp_id, exp in self.experiments.items()
        }
        with open(experiments_file, 'w') as f:
            json.dump(experiments_dict, f, indent=2, default=str)
        
        # Save interactions
        if self.interactions:
            interactions_file = os.path.join(self.data_dir, 'interactions.csv')
            df = pd.DataFrame([i.to_dict() for i in self.interactions])
            df.to_csv(interactions_file, index=False)
        
        # Save user assignments
        assignments_file = os.path.join(self.data_dir, 'user_assignments.json')
        with open(assignments_file, 'w') as f:
            json.dump(self.user_assignments, f, indent=2)
    
    def create_experiment(self, experiment_id: str, name: str, description: str,
                         variants: List[str], traffic_split: Dict[str, float],
                         metrics: List[str], duration_days: int = 30) -> Experiment:
        """
        Create a new A/B testing experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable name
            description: Description of what's being tested
            variants: List of variant names (e.g., ['control', 'variant_a'])
            traffic_split: Dict mapping variant to traffic percentage (must sum to 1.0)
            metrics: List of metrics to track
            duration_days: How long the experiment should run
            
        Returns:
            Created Experiment object
        """
        if sum(traffic_split.values()) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")
        
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            status='active',
            traffic_split=traffic_split,
            metrics=metrics
        )
        
        self.experiments[experiment_id] = experiment
        self._save_data()
        return experiment
    
    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> str:
        """
        Assign a user to a variant for an experiment.
        Uses consistent hashing to ensure same user always gets same variant.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            
        Returns:
            Variant name assigned to user
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Check if user already assigned
        if user_id in self.user_assignments and experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]
        
        # Consistent assignment using hash
        hash_value = hash(f"{user_id}_{experiment_id}") % 10000
        cumulative = 0
        assigned_variant = None
        
        for variant, percentage in experiment.traffic_split.items():
            cumulative += percentage * 10000
            if hash_value < cumulative:
                assigned_variant = variant
                break
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = assigned_variant
        
        self._save_data()
        return assigned_variant
    
    def track_interaction(self, user_id: str, experiment_id: str, variant: str,
                         interaction_type: str, recommended_activities: List[str],
                         clicked_activity: Optional[str] = None,
                         booking_confirmed: bool = False,
                         session_id: Optional[str] = None):
        """
        Track a user interaction with recommendations.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            variant: Variant the user was assigned to
            interaction_type: Type of interaction ('impression', 'click', 'booking', 'skip')
            recommended_activities: List of activities that were recommended
            clicked_activity: Activity that was clicked (if any)
            booking_confirmed: Whether a booking was confirmed
            session_id: Optional session identifier
        """
        interaction = UserInteraction(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            recommended_activities=recommended_activities,
            clicked_activity=clicked_activity,
            booking_confirmed=booking_confirmed,
            session_id=session_id
        )
        
        self.interactions.append(interaction)
        self._save_data()
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each variant in an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dict mapping variant to metrics
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        variant_metrics = {variant: defaultdict(int) for variant in experiment.variants}
        
        # Filter interactions for this experiment
        exp_interactions = [
            i for i in self.interactions 
            if i.experiment_id == experiment_id
        ]
        
        # Group by variant
        for interaction in exp_interactions:
            variant = interaction.variant
            variant_metrics[variant]['total_interactions'] += 1
            
            if interaction.interaction_type == 'impression':
                variant_metrics[variant]['impressions'] += 1
            elif interaction.interaction_type == 'click':
                variant_metrics[variant]['clicks'] += 1
            elif interaction.interaction_type == 'booking':
                variant_metrics[variant]['bookings'] += 1
                if interaction.booking_confirmed:
                    variant_metrics[variant]['confirmed_bookings'] += 1
        
        # Calculate derived metrics
        for variant in experiment.variants:
            metrics = variant_metrics[variant]
            if metrics['impressions'] > 0:
                metrics['click_through_rate'] = metrics['clicks'] / metrics['impressions']
            else:
                metrics['click_through_rate'] = 0.0
            
            if metrics['clicks'] > 0:
                metrics['conversion_rate'] = metrics['bookings'] / metrics['clicks']
            else:
                metrics['conversion_rate'] = 0.0
            
            if metrics['impressions'] > 0:
                metrics['booking_rate'] = metrics['bookings'] / metrics['impressions']
            else:
                metrics['booking_rate'] = 0.0
            
            # Engagement score (weighted combination)
            metrics['engagement_score'] = (
                metrics['click_through_rate'] * 0.4 +
                metrics['conversion_rate'] * 0.4 +
                metrics['booking_rate'] * 0.2
            )
        
        return {variant: dict(metrics) for variant, metrics in variant_metrics.items()}
    
    def compare_variants(self, experiment_id: str) -> Dict[str, any]:
        """
        Compare variants and determine statistical significance.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Comparison results with statistical tests
        """
        metrics = self.get_experiment_metrics(experiment_id)
        experiment = self.experiments[experiment_id]
        
        # Get interaction data for statistical testing
        exp_interactions = [
            i for i in self.interactions 
            if i.experiment_id == experiment_id
        ]
        
        results = {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'variants': {},
            'comparison': {}
        }
        
        # Calculate metrics for each variant
        for variant in experiment.variants:
            variant_interactions = [i for i in exp_interactions if i.variant == variant]
            results['variants'][variant] = {
                'metrics': metrics[variant],
                'sample_size': len(variant_interactions)
            }
        
        # Compare variants (simple comparison for now)
        if len(experiment.variants) >= 2:
            control_variant = experiment.variants[0]
            control_metrics = metrics[control_variant]
            
            for variant in experiment.variants[1:]:
                variant_metrics = metrics[variant]
                comparison = {
                    'control': control_variant,
                    'variant': variant,
                    'ctr_improvement': (
                        (variant_metrics['click_through_rate'] - control_metrics['click_through_rate']) /
                        control_metrics['click_through_rate'] * 100
                        if control_metrics['click_through_rate'] > 0 else 0
                    ),
                    'conversion_improvement': (
                        (variant_metrics['conversion_rate'] - control_metrics['conversion_rate']) /
                        control_metrics['conversion_rate'] * 100
                        if control_metrics['conversion_rate'] > 0 else 0
                    ),
                    'engagement_improvement': (
                        (variant_metrics['engagement_score'] - control_metrics['engagement_score']) /
                        control_metrics['engagement_score'] * 100
                        if control_metrics['engagement_score'] > 0 else 0
                    )
                }
                results['comparison'][variant] = comparison
        
        return results
    
    def get_recommendation_strategy(self, user_id: str, experiment_id: str) -> str:
        """
        Get the recommendation strategy (variant) for a user.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            
        Returns:
            Variant name
        """
        return self.assign_user_to_variant(user_id, experiment_id)
    
    def list_active_experiments(self) -> List[Experiment]:
        """Get list of all active experiments."""
        return [
            exp for exp in self.experiments.values()
            if exp.status == 'active' and 
            (exp.end_date is None or exp.end_date > datetime.now())
        ]

