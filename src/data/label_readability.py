import openai
import csv
import os
import time
from datasets import load_dataset
import json
import re
from typing import Tuple, Optional

class ReadabilityLabeler:
    """Enhanced readability labeling system with better error handling and progress tracking."""
    
    def __init__(self, api_key: str, output_file: str = "src/data/readability_dataset.csv"):
        """
        Initialize the readability labeler.
        
        Args:
            api_key: OpenAI API key
            output_file: Path to output CSV file
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.output_file = output_file
        self.processed_count = 0
        self.error_count = 0
        self.total_cost = 0.0
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    def get_readability_score(self, code_snippet: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Get readability score from GPT-4 with better error handling.
        
        Args:
            code_snippet: Python code to analyze
            
        Returns:
            Tuple of (score, reason, raw_response) or (None, None, error_msg)
        """
        # Clean and truncate code snippet
        cleaned_code = code_snippet.strip()
        if len(cleaned_code) > 2000:  # Limit to avoid huge API costs
            cleaned_code = cleaned_code[:2000] + "..."
            
        prompt = f"""
Rate the readability of this Python function on a scale of 1-5:
1 = Very hard to read (poor naming, no structure, confusing)
2 = Hard to read (some issues with clarity)
3 = Moderately readable (decent but could be improved)
4 = Good readability (clear and well-structured)
5 = Excellent readability (perfect naming, structure, comments)

Code:
```python
{cleaned_code}
```

Respond ONLY in this exact format:
Score: [1-5]
Reason: [Brief explanation in one sentence]
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for labeling
                messages=[
                    {"role": "system", "content": "You are a code readability expert. Rate code objectively and consistently."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=100    # Limit response length
            )

            content = response.choices[0].message.content.strip()
            
            # Estimate cost (rough calculation)
            input_tokens = len(prompt) // 4  # Rough token estimation
            output_tokens = len(content) // 4
            cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000  # GPT-4o-mini pricing
            self.total_cost += cost
            
            # Parse the response
            score, reason = self._parse_response(content)
            return score, reason, content
            
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            print(f"ERROR: {error_msg}")
            return None, None, error_msg
    
    def _parse_response(self, response: str) -> Tuple[Optional[int], Optional[str]]:
        """Parse GPT-4 response to extract score and reason."""
        try:
            # Look for "Score: X" pattern
            score_match = re.search(r'Score:\s*([1-5])', response)
            reason_match = re.search(r'Reason:\s*(.+)', response, re.IGNORECASE)
            
            if score_match and reason_match:
                score = int(score_match.group(1))
                reason = reason_match.group(1).strip()
                return score, reason
            else:
                # Fallback: try to extract just a number
                numbers = re.findall(r'\b([1-5])\b', response)
                if numbers:
                    return int(numbers[0]), response.strip()
                else:
                    return None, None
                    
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            return None, None
    
    def create_dataset(self, num_samples: int = 1000, test_mode: bool = False):
        """
        Create the readability dataset.
        
        Args:
            num_samples: Number of samples to process
            test_mode: If True, only process 10 samples for testing
        """
        if test_mode:
            num_samples = 10
            print("Test mode active, processing only 10 samples")
        
        print(f"Starting dataset creation with {num_samples} samples...")
        print(f"Estimated cost: ${num_samples * 0.01:.2f} - ${num_samples * 0.02:.2f}")
        
        try:
            # Load CodeSearchNet dataset
            print("Loading CodeSearchNet dataset...")
            dataset = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
            print(f"Dataset loaded with {len(dataset['train'])} training samples")
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return False
        
        # Prepare CSV file
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["code_snippet", "readability_score", "reason", "tokens", "lines"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            print(f"Processing samples...")
            start_time = time.time()
            
            for i, sample in enumerate(dataset["train"][:num_samples]):
                if i % 50 == 0:  # Progress update every 50 samples
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (num_samples - i) / rate if rate > 0 else 0
                    print(f"Progress: {i}/{num_samples} ({i/num_samples*100:.1f}%) - "
                          f"Cost: ${self.total_cost:.3f} - ETA: {eta/60:.1f}min")
                
                # Extract code
                if isinstance(sample, dict) and "func_code_string" in sample:
                    code_snippet = sample["func_code_string"]
                    
                    if not code_snippet or len(code_snippet.strip()) < 20:
                        continue  # Skip empty or very short code
                    
                    # Get readability score
                    score, reason, raw_response = self.get_readability_score(code_snippet)
                    
                    if score is not None:
                        # Calculate additional metrics
                        tokens = len(code_snippet.split())
                        lines = len(code_snippet.split('\n'))
                        
                        writer.writerow({
                            "code_snippet": code_snippet.replace('\n', '\\n'),  # Escape newlines for CSV
                            "readability_score": score,
                            "reason": reason or "No reason provided",
                            "tokens": tokens,
                            "lines": lines
                        })
                        
                        self.processed_count += 1
                        if i % 10 == 0:
                            print(f"Processed sample {i+1}: Score {score} - {reason[:50]}...")
                    else:
                        self.error_count += 1
                        print(f"⚠️  Skipped sample {i+1} due to error")
                    
                    # Rate limiting - small delay to avoid hitting API limits
                    time.sleep(0.1)
                
                else:
                    print(f"⚠️  Sample {i+1}: No code found")
                    continue
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\nDataset creation completed!")
        print(f"Statistics:")
        print(f"   • Total processed: {self.processed_count}")
        print(f"   • Errors: {self.error_count}")
        print(f"   • Success rate: {self.processed_count/(self.processed_count+self.error_count)*100:.1f}%")
        print(f"   • Total cost: ${self.total_cost:.3f}")
        print(f"   • Time taken: {total_time/60:.1f} minutes")
        print(f"   • Output file: {self.output_file}")
        
        return True

def main():
    """Main function to run the labeling process."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API Key is missing.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("=" * 50)
    
    # Ask user for preferences
    test_mode = input("Run in test mode (10 samples)? [y/N]: ").lower().startswith('y')
    
    if not test_mode:
        num_samples = input("How many samples to process? [1000]: ")
        try:
            num_samples = int(num_samples) if num_samples else 1000
        except ValueError:
            num_samples = 1000
            
        confirm = input(f"This will cost ~${num_samples * 0.015:.2f}. Continue? [y/N]: ")
        if not confirm.lower().startswith('y'):
            print("Cancelled by user")
            return
    else:
        num_samples = 10
    
    # Create labeler and run
    labeler = ReadabilityLabeler(api_key)
    success = labeler.create_dataset(num_samples, test_mode)
    
    if success:
        print(f"\nDataset ready for training!")
    else:
        print(f"\nDataset creation failed")

if __name__ == "__main__":
    main()