from feedback.collector import FeedbackCollector
import os
from datetime import datetime, timedelta

class ImprovementMetricsReporter:
    """
    Generates reports on system improvement based on user feedback.
    """
    
    def __init__(self, output_dir="reports"):
        """
        Initialize the metrics reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.feedback_collector = FeedbackCollector()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_improvement_report(self):
        """
        Generate a comprehensive improvement report.
        
        Returns:
            Path to the generated report
        """
        metrics = self.feedback_collector.get_metrics()
        
        # Format timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"improvement_report_{timestamp}.txt")
        
        # Check if we have enough data
        if metrics["total_suggestions"] == 0:
            with open(report_file, 'w') as f:
                f.write("=== COGNITO IMPROVEMENT METRICS REPORT ===\n\n")
                f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("No feedback data available yet. Use Cognito and provide feedback on suggestions to build metrics.\n\n")
                f.write("For your CV:\n")
                f.write("• Implemented a feedback-driven learning system for code analysis\n")
                f.write("• Designed an adaptive suggestion system with continuous improvement capabilities\n")
                f.write("• Created a metrics reporting system for tracking code quality improvement over time\n")
            return report_file
            
        # If we have data, calculate real metrics
        weekly_improvement = self.feedback_collector.get_improvement_metrics(interval_days=7)
        monthly_improvement = self.feedback_collector.get_improvement_metrics(interval_days=30)
        quarterly_improvement = self.feedback_collector.get_improvement_metrics(interval_days=90)
        
        with open(report_file, 'w') as f:
            f.write("=== COGNITO IMPROVEMENT METRICS REPORT ===\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== OVERALL METRICS ===\n")
            f.write(f"Total suggestions provided: {metrics['total_suggestions']}\n")
            f.write(f"Accepted suggestions: {metrics['accepted_suggestions']} ({metrics['acceptance_rate']}%)\n")
            f.write(f"Rejected suggestions: {metrics['rejected_suggestions']} ({100 - metrics['acceptance_rate']}%)\n\n")
            
            # Category performance
            f.write("=== CATEGORY PERFORMANCE ===\n")
            for category in ["Security", "Readability", "Performance", "Best Practices"]:
                perf = self.feedback_collector.get_suggestion_performance(category)
                if perf["total"] > 0:
                    f.write(f"{category}: {perf['acceptance_rate']}% acceptance rate ({perf['accepted']}/{perf['total']})\n")
            f.write("\n")
            
            # Improvement over time
            f.write("=== IMPROVEMENT OVER TIME ===\n")
            
            if weekly_improvement["previous_period"]["total"] > 0:
                f.write(f"Weekly improvement: {weekly_improvement['acceptance_improvement_percentage']}%\n")
                f.write(f"  Current week: {weekly_improvement['current_period']['acceptance_rate']}% acceptance\n")
                f.write(f"  Previous week: {weekly_improvement['previous_period']['acceptance_rate']}% acceptance\n\n")
            
            if monthly_improvement["previous_period"]["total"] > 0:
                f.write(f"Monthly improvement: {monthly_improvement['acceptance_improvement_percentage']}%\n")
                f.write(f"  Current month: {monthly_improvement['current_period']['acceptance_rate']}% acceptance\n")
                f.write(f"  Previous month: {monthly_improvement['previous_period']['acceptance_rate']}% acceptance\n\n")
            
            if quarterly_improvement["previous_period"]["total"] > 0:
                f.write(f"Quarterly improvement: {quarterly_improvement['acceptance_improvement_percentage']}%\n")
                f.write(f"  Current quarter: {quarterly_improvement['current_period']['acceptance_rate']}% acceptance\n")
                f.write(f"  Previous quarter: {quarterly_improvement['previous_period']['acceptance_rate']}% acceptance\n\n")
            
            # CV metrics
            # f.write("=== CV-WORTHY METRICS ===\n")
            # if metrics["total_suggestions"] > 0:
            #     f.write(f"• Processed {metrics['total_suggestions']} code suggestions with {metrics['acceptance_rate']}% acceptance rate\n")
            
            # if monthly_improvement["previous_period"]["total"] > 0:
            #     f.write(f"• Improved suggestion acceptance rate by {monthly_improvement['acceptance_improvement_percentage']}% over one month\n")
            
            # # Calculate improvement in specific categories
            # cv_metrics = []
            # for category in ["Security", "Readability", "Performance"]:
            #     perf = self.feedback_collector.get_suggestion_performance(category)
            #     if perf["total"] > 5:
            #         cv_metrics.append(f"• {category} suggestion acceptance rate: {perf['acceptance_rate']}%\n")
            
            # for metric in cv_metrics:
            #     f.write(metric)
            
            # if len(cv_metrics) == 0:
            #     f.write("• Implemented feedback-driven learning system to improve code review quality\n")
            #     f.write("• Developed adaptive AI system that continuously improves from user feedback\n")
            
        return report_file