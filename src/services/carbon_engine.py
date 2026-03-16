from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

class CarbonEngine:
    """
    EcoLife Carbon Management Decision Engine.
    Handles gamification (Carbon Credits) and personalized recommendations.
    """

    def __init__(self, baseline_kg: float = 10.0, reward_factor: float = 5.0):
        self.baseline_kg = baseline_kg # Standard daily baseline (e.g., global average)
        self.reward_factor = reward_factor

    def calculate_credits(self, actual_kg: float, predicted_kg: float) -> Dict[str, float]:
        """
        Calculate Carbon Credits earned based on performance against baseline and prediction.
        
        Args:
            actual_kg: The verified carbon footprint for the period.
            predicted_kg: The AI-predicted footprint (personal goal).
            
        Returns:
            Dict breakdown of credits.
        """
        base_credits = 0.0
        bonus_credits = 0.0
        
        # 1. Baseline Reduction Reward
        if actual_kg < self.baseline_kg:
            base_credits = (self.baseline_kg - actual_kg) * self.reward_factor
        
        # 2. Prediction Beat Bonus (Gamification)
        # Encourages beating your own AI forecast
        if actual_kg < predicted_kg:
            bonus_credits = (predicted_kg - actual_kg) * (self.reward_factor * 1.5)
            
        total_credits = max(0, base_credits + bonus_credits)
        
        return {
            "total_credits": round(total_credits, 2),
            "base_credits": round(base_credits, 2),
            "bonus_credits": round(bonus_credits, 2),
            "carbon_saved_vs_baseline": round(max(0, self.baseline_kg - actual_kg), 2)
        }

    def generate_recommendations(self, user_features: Dict[str, Any]) -> List[str]:
        """
        Generate personalized green lifestyle recommendations based on feature vector.
        
        Args:
            user_features: Dictionary of user behavior (e.g., transport_dist, diet_type).
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Example Logic (needs real feature names from data)
        # Assuming keys like 'transport_distance', 'vehicle_type', 'diet'
        
        dist = user_features.get('Vehicle Distance Km', 0)
        if dist > 20:
            recommendations.append("Consider carpooling or public transit for long commutes to save ~5kg CO2.")
        
        diet = user_features.get('Diet Type', 'omnivores')
        if str(diet).lower() in ['omnivores', 'high meat']:
            recommendations.append("Trying a 'Meatless Monday' can reduce your weekly footprint by 15%.")
            
        energy = user_features.get('Heating Energy Source', 'coal')
        if 'coal' in str(energy).lower() or 'oil' in str(energy).lower():
            recommendations.append("Switching to renewable energy providers or improving insulation is a high-impact move.")
            
        if not recommendations:
            recommendations.append("Great job maintaining low emissions! Keep monitoring your dashboard.")
            
        return recommendations

    def evaluate_performance(self, actual: float, predicted: float, features: Dict[str, Any]) -> Dict[str, Any]:
        """Full audit cycle: Credits + Recommendations."""
        credits = self.calculate_credits(actual, predicted)
        recs = self.generate_recommendations(features)
        
        return {
            "credits": credits,
            "recommendations": recs,
            "status": "Green Star" if actual < self.baseline_kg * 0.8 else "Neutral"
        }
