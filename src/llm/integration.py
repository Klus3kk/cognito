"""
Integration module for connecting LLM capabilities with the core analyzer.
"""

from typing import Dict, Any
from .code_assistant import CodeAssistant

class LLMIntegration:
    """Provides integration between Cognito analyzers and LLM capabilities."""
    
    def __init__(self):
        """Initialize the LLM integration."""
        self.code_assistant = CodeAssistant()
    
    def enhance_analysis(self, code: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance analysis results with LLM-generated insights.
        
        Args:
            code: The code being analyzed
            analysis_results: Original Cognito analysis results
            
        Returns:
            Enhanced analysis results with LLM insights
        """
        # Skip if LLM features aren't available
        if not self.code_assistant.llm_available:
            analysis_results["llm_enhanced"] = False
            return analysis_results
            
        # Add explanation of the code
        explanation = self.code_assistant.explain_code(code)
        
        # Format analysis results for the LLM
        formatted_results = self._format_results(analysis_results)
        
        # Get AI-powered review
        review = self.code_assistant.review_code(code, formatted_results)
        
        # Add LLM enhancements to results
        analysis_results["llm_enhanced"] = True
        analysis_results["code_explanation"] = explanation
        analysis_results["ai_review"] = review
        
        return analysis_results
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results as a string for LLM consumption."""
        formatted = []
        
        if "summary" in results:
            formatted.append("Summary:")
            for key, value in results["summary"].items():
                formatted.append(f"- {key}: {value}")
        
        if "suggestions" in results:
            formatted.append("\nSuggestions:")
            for suggestion in results["suggestions"]:
                category = suggestion.get("category", "")
                message = suggestion.get("message", "")
                priority = suggestion.get("priority", "")
                formatted.append(f"- [{priority}] {category}: {message}")
        
        return "\n".join(formatted)