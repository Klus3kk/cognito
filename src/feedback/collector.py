import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class FeedbackCollector:
    """
    Collects and manages user feedback on suggestions to improve model accuracy over time.
    """
    
    def __init__(self, feedback_file: str = "data/feedback.json"):
        """
        Initialize the feedback collector.
        
        Args:
            feedback_file: Path to the JSON file storing feedback data
        """
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
        self.feedback_data = self._load_feedback()
    
    def _ensure_feedback_file(self) -> None:
        """Ensure the feedback file and directory exist."""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump({
                    "suggestions": [],
                    "metrics": {
                        "total_suggestions": 0,
                        "accepted_suggestions": 0,
                        "rejected_suggestions": 0,
                        "acceptance_rate": 0.0
                    },
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
    
    def _load_feedback(self) -> Dict[str, Any]:
        """Load feedback data from the JSON file."""
        with open(self.feedback_file, 'r') as f:
            return json.load(f)
    
    def _save_feedback(self) -> None:
        """Save feedback data to the JSON file."""
        self.feedback_data["last_updated"] = datetime.now().isoformat()
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_feedback(self, suggestion: Dict[str, Any], accepted: bool, user_comment: Optional[str] = None) -> None:
        """
        Add feedback for a suggestion.
        
        Args:
            suggestion: The suggestion that was presented
            accepted: Whether the user accepted the suggestion
            user_comment: Optional comment from the user
        """
        feedback_entry = {
            "suggestion": suggestion,
            "accepted": accepted,
            "timestamp": datetime.now().isoformat(),
            "user_comment": user_comment
        }
        
        self.feedback_data["suggestions"].append(feedback_entry)
        self.feedback_data["metrics"]["total_suggestions"] += 1
        
        if accepted:
            self.feedback_data["metrics"]["accepted_suggestions"] += 1
        else:
            self.feedback_data["metrics"]["rejected_suggestions"] += 1
        
        # Update acceptance rate
        total = self.feedback_data["metrics"]["total_suggestions"]
        accepted_count = self.feedback_data["metrics"]["accepted_suggestions"]
        self.feedback_data["metrics"]["acceptance_rate"] = round((accepted_count / total) * 100, 2) if total > 0 else 0.0
        
        self._save_feedback()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current feedback metrics."""
        return self.feedback_data["metrics"]
    
    def get_suggestion_performance(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for suggestions, optionally filtered by category.
        
        Args:
            category: Optional category to filter suggestions
            
        Returns:
            Dictionary of performance metrics
        """
        suggestions = self.feedback_data["suggestions"]
        
        if category:
            suggestions = [s for s in suggestions if s["suggestion"].get("category") == category]
        
        total = len(suggestions)
        if total == 0:
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "acceptance_rate": 0.0
            }
        
        accepted = sum(1 for s in suggestions if s["accepted"])
        
        return {
            "total": total,
            "accepted": accepted,
            "rejected": total - accepted,
            "acceptance_rate": round((accepted / total) * 100, 2) if total > 0 else 0.0
        }
    
    def get_improvement_metrics(self, interval_days: int = 30) -> Dict[str, Any]:
        """
        Calculate improvement metrics over time.
        
        Args:
            interval_days: Number of days to use for comparison
            
        Returns:
            Dictionary with improvement metrics
        """
        now = datetime.now()
        
        # Filter suggestions by time periods
        current_period = [
            s for s in self.feedback_data["suggestions"] 
            if (now - datetime.fromisoformat(s["timestamp"])).days <= interval_days
        ]
        
        previous_period = [
            s for s in self.feedback_data["suggestions"] 
            if interval_days < (now - datetime.fromisoformat(s["timestamp"])).days <= interval_days * 2
        ]
        
        # Calculate metrics for current period
        current_total = len(current_period)
        current_accepted = sum(1 for s in current_period if s["accepted"])
        current_rate = (current_accepted / current_total * 100) if current_total > 0 else 0
        
        # Calculate metrics for previous period
        previous_total = len(previous_period)
        previous_accepted = sum(1 for s in previous_period if s["accepted"])
        previous_rate = (previous_accepted / previous_total * 100) if previous_total > 0 else 0
        
        # Calculate improvement (avoid division by zero)
        acceptance_improvement = current_rate - previous_rate
        
        return {
            "current_period": {
                "total": current_total,
                "accepted": current_accepted,
                "acceptance_rate": round(current_rate, 2)
            },
            "previous_period": {
                "total": previous_total,
                "accepted": previous_accepted,
                "acceptance_rate": round(previous_rate, 2)
            },
            "acceptance_improvement": round(acceptance_improvement, 2),
            "acceptance_improvement_percentage": round(
                (acceptance_improvement / previous_rate * 100) if previous_rate > 0 else 0, 2
            )
        }