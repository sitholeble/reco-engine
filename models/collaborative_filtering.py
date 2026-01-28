"""
Simple Collaborative Filtering Model (Amazon-style)
"People who did X also did Y"
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    """
    Simple collaborative filtering model using item-based similarity.
    Similar to Amazon's "customers who bought this item also bought..."
    """
    
    def __init__(self):
        self.user_activity_matrix = None
        self.activity_similarity = None
        self.activity_to_index = {}
        self.index_to_activity = {}
        self.user_to_index = {}
        self.index_to_user = {}
        
    def fit(self, interaction_data):
        """
        Fit the model on interaction data.
        
        Args:
            interaction_data: DataFrame with columns ['user_id', 'activity', 'count'] or similar
        """
        # Create user-activity matrix
        pivot_data = interaction_data.pivot_table(
            index='user_id',
            columns='activity',
            values='count' if 'count' in interaction_data.columns else 'avg_rating',
            fill_value=0
        )
        
        # Store mappings
        self.activity_to_index = {act: idx for idx, act in enumerate(pivot_data.columns)}
        self.index_to_activity = {idx: act for act, idx in self.activity_to_index.items()}
        self.user_to_index = {user: idx for idx, user in enumerate(pivot_data.index)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}
        
        # Convert to sparse matrix for efficiency
        self.user_activity_matrix = csr_matrix(pivot_data.values)
        
        # Compute activity-to-activity similarity (cosine similarity)
        # Transpose to get activity-activity similarity
        activity_matrix = self.user_activity_matrix.T
        self.activity_similarity = cosine_similarity(activity_matrix, dense_output=False)
        
    def recommend_similar_activities(self, activity, top_k=5):
        """
        Find activities similar to the given activity.
        "People who did X also did Y"
        
        Args:
            activity: Activity name
            top_k: Number of recommendations
            
        Returns:
            List of tuples (activity, similarity_score)
        """
        if activity not in self.activity_to_index:
            return []
        
        activity_idx = self.activity_to_index[activity]
        similarities = self.activity_similarity[activity_idx].toarray().flatten()
        
        # Get top k similar activities (excluding the activity itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        recommendations = [
            (self.index_to_activity[idx], similarities[idx])
            for idx in top_indices
            if similarities[idx] > 0
        ]
        
        return recommendations
    
    def recommend_for_user(self, user_id, top_k=5):
        """
        Recommend activities for a user based on their history.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            
        Returns:
            List of tuples (activity, score)
        """
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        user_vector = self.user_activity_matrix[user_idx].toarray().flatten()
        
        # Find activities user hasn't done yet
        done_activities = set(np.where(user_vector > 0)[0])
        all_activities = set(range(len(self.index_to_activity)))
        candidate_activities = all_activities - done_activities
        
        if not candidate_activities:
            return []
        
        # Score each candidate activity based on similarity to user's done activities
        scores = {}
        for candidate_idx in candidate_activities:
            score = 0
            for done_idx in done_activities:
                similarity = self.activity_similarity[candidate_idx, done_idx]
                user_weight = user_vector[done_idx]
                score += similarity * user_weight
            scores[candidate_idx] = score
        
        # Get top k recommendations
        top_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        recommendations = [
            (self.index_to_activity[idx], score)
            for idx, score in top_indices
            if score > 0
        ]
        
        return recommendations
    
    def get_popular_activities(self, top_k=10):
        """Get most popular activities overall."""
        activity_counts = np.array(self.user_activity_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(activity_counts)[::-1][:top_k]
        
        return [
            (self.index_to_activity[idx], activity_counts[idx])
            for idx in top_indices
        ]

