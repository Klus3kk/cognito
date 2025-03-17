# src/llm/learning_enhancer.py
from typing import Dict, Any, List
import os
import json
from datetime import datetime
from feedback.collector import FeedbackCollector
from llm.integration import LLMIntegration

class LearningLLMIntegration(LLMIntegration):
    """
    Enhanced LLM integration that learns from user feedback over time.
    """
    
    def __init__(self):
        """Initialize the learning LLM integration with feedback collection."""
        super().__init__()
        self.feedback_collector = FeedbackCollector()
        self.adaptation_prompts = self._generate_adaptation_prompts()
    
    def _generate_adaptation_prompts(self) -> Dict[str, str]:
        """
        Generate adaptation prompts based on feedback data.
        These prompts are injected into the LLM to guide its responses.
        """
        metrics = self.feedback_collector.get_metrics()
        
        # Get category-specific metrics
        security_metrics = self.feedback_collector.get_suggestion_performance("Security")
        readability_metrics = self.feedback_collector.get_suggestion_performance("Readability")
        performance_metrics = self.feedback_collector.get_suggestion_performance("Performance")
        
        # Create adaptation prompts based on metrics
        prompts = {}
        
        # General adaptation prompt
        prompts["general"] = f"""
        Based on user feedback, {metrics['acceptance_rate']}% of suggestions are accepted.
        Focus more on actionable, specific improvements that can be implemented easily.
        """
        
        # Category-specific adaptation prompts
        if security_metrics["total"] > 0:
            acceptance = security_metrics["acceptance_rate"]
            if acceptance < 50:
                prompts["security"] = f"""
                Security suggestions have a low acceptance rate ({acceptance}%).
                Focus on critical vulnerabilities with clear exploit paths.
                Be more specific about the risks and provide concrete examples.
                """
            else:
                prompts["security"] = f"""
                Security suggestions have a good acceptance rate ({acceptance}%).
                Continue focusing on OWASP top 10 vulnerabilities and providing clear remediation steps.
                """
                
        if readability_metrics["total"] > 0:
            acceptance = readability_metrics["acceptance_rate"]
            if acceptance < 50:
                prompts["readability"] = f"""
                Readability suggestions have a low acceptance rate ({acceptance}%).
                Focus on substantive issues like function length and complexity rather than stylistic preferences.
                Provide specific examples of how to improve readability.
                """
            else:
                prompts["readability"] = f"""
                Readability suggestions have a good acceptance rate ({acceptance}%).
                Continue focusing on naming conventions, documentation, and code structure.
                """
                
        if performance_metrics["total"] > 0:
            acceptance = performance_metrics["acceptance_rate"]
            if acceptance < 50:
                prompts["performance"] = f"""
                Performance suggestions have a low acceptance rate ({acceptance}%).
                Focus on algorithmic complexity issues with clear performance impacts.
                Provide benchmarks or complexity analysis when suggesting optimizations.
                """
            else:
                prompts["performance"] = f"""
                Performance suggestions have a good acceptance rate ({acceptance}%).
                Continue focusing on algorithmic efficiency and resource usage.
                """
        
        return prompts
    
    def enhance_analysis(self, code: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance analysis results with LLM-generated insights, using feedback to improve suggestions.
        
        Args:
            code: The code being analyzed
            analysis_results: Original Cognito analysis results
            
        Returns:
            Enhanced analysis results with adaptive LLM insights
        """
        # Skip if LLM features aren't available
        if not self.code_assistant.llm_available:
            analysis_results["llm_enhanced"] = False
            return analysis_results
        
        # Add adaptation insights to the analysis
        adaptation_insights = self._get_adaptation_insights()
        
        # Format analysis results for the LLM
        formatted_results = self._format_results(analysis_results)
        
        # Get AI-powered review with adaptation prompts
        review_prompt = f"""
        {self.adaptation_prompts.get('general', '')}
        
        When reviewing security issues:
        {self.adaptation_prompts.get('security', '')}
        
        When reviewing readability issues:
        {self.adaptation_prompts.get('readability', '')}
        
        When reviewing performance issues:
        {self.adaptation_prompts.get('performance', '')}
        
        Here is the analysis to review:
        {formatted_results}
        """
        
        review = self.code_assistant.review_code(code, review_prompt)
        explanation = self.code_assistant.explain_code(code)
        
        # Add LLM enhancements to results
        analysis_results["llm_enhanced"] = True
        analysis_results["code_explanation"] = explanation
        analysis_results["ai_review"] = review
        analysis_results["adaptation_insights"] = adaptation_insights
        
        return analysis_results
    
    def _get_adaptation_insights(self) -> Dict[str, Any]:
        """
        Generate insights about how the system is adapting based on feedback.
        
        Returns:
            Dictionary of adaptation insights
        """
        # Get improvement metrics for different time periods
        monthly_improvement = self.feedback_collector.get_improvement_metrics(interval_days=30)
        weekly_improvement = self.feedback_collector.get_improvement_metrics(interval_days=7)
        
        return {
            "monthly_improvement": monthly_improvement,
            "weekly_improvement": weekly_improvement,
            "adaptation_active": True,
            "total_suggestions_processed": self.feedback_collector.get_metrics()["total_suggestions"]
        }
    
    def add_suggestion_feedback(self, suggestion: Dict[str, Any], accepted: bool, comment: str = None) -> None:
        """
        Add feedback for a specific suggestion.
        
        Args:
            suggestion: The suggestion that was presented
            accepted: Whether the user accepted the suggestion
            comment: Optional comment about why the suggestion was accepted or rejected
        """
        self.feedback_collector.add_feedback(suggestion, accepted, comment)
        # Regenerate adaptation prompts after new feedback
        self.adaptation_prompts = self._generate_adaptation_prompts()