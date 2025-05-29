import openai
import csv
import os
import time
import requests
import json
import re
from typing import Tuple, Optional, List

class DatasetLoader:
    """Load real Python code by directly accessing GitHub and other sources."""
    
    def __init__(self, api_key: str = None, output_file: str = "src/data/readability_dataset.csv"):
        """Initialize the dataset loader."""
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.output_file = output_file
        self.processed_count = 0
        self.error_count = 0
        self.total_cost = 0.0
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def load_real_datasets(self, num_samples: int = 1000, test_mode: bool = False):
        """
        Load real Python code from GitHub and other direct sources.
        
        Args:
            num_samples: Number of samples to collect
            test_mode: If True, only collect 10 samples for testing
        """
        if test_mode:
            num_samples = 10
            print("Test mode: collecting only 10 samples")
        
        print(f"Loading real Python code from direct sources...")
        
        all_code_samples = []
        
        # Source 1: Direct GitHub - HumanEval
        print("\n1️⃣ Loading from GitHub - HumanEval...")
        samples_1 = self._load_from_github_humaneval(num_samples // 4)
        all_code_samples.extend(samples_1)
        print(f"   ✅ Collected {len(samples_1)} samples from GitHub HumanEval")
        
        # Source 2: Direct GitHub - Popular Python repos
        print("\n2️⃣ Loading from GitHub - Popular Python files...")
        samples_2 = self._load_from_github_repos(num_samples // 4)
        all_code_samples.extend(samples_2)
        print(f"   ✅ Collected {len(samples_2)} samples from GitHub repos")
        
        # Source 3: Python Examples from Raw URLs
        print("\n3️⃣ Loading Python examples from raw sources...")
        samples_3 = self._load_from_raw_sources(num_samples // 4)
        all_code_samples.extend(samples_3)
        print(f"   ✅ Collected {len(samples_3)} samples from raw sources")
        
        # Source 4: Built-in Quality Examples
        print("\n4️⃣ Loading built-in quality examples...")
        samples_4 = self._get_quality_examples(num_samples // 4)
        all_code_samples.extend(samples_4)
        print(f"   ✅ Collected {len(samples_4)} samples from quality examples")
        
        print(f"\n📊 Total collected: {len(all_code_samples)} real code samples")
        
        # If we still need more, add local examples
        if len(all_code_samples) < num_samples:
            needed = num_samples - len(all_code_samples)
            print(f"   Adding {needed} local examples to reach target...")
            local_samples = self._get_local_examples()[:needed]
            all_code_samples.extend(local_samples)
        
        # Shuffle and take only what we need
        import random
        random.shuffle(all_code_samples)
        final_samples = all_code_samples[:num_samples]
        
        print(f"📝 Final dataset: {len(final_samples)} samples")
        
        # Now label them (if API key provided) or use heuristic scoring
        if self.client:
            return self._label_with_openai(final_samples)
        else:
            return self._label_with_heuristics(final_samples)
    
    def _load_from_github_humaneval(self, max_samples: int) -> List[str]:
        """Load HumanEval data directly from GitHub."""
        samples = []
        try:
            print("   Fetching HumanEval from GitHub...")
            
            # Direct access to HumanEval JSON data
            url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                count = 0
                
                for line in lines:
                    if count >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line)
                        prompt = data.get('prompt', '')
                        canonical_solution = data.get('canonical_solution', '')
                        
                        if prompt and canonical_solution:
                            full_code = prompt + canonical_solution
                            if self._is_good_code_sample(full_code):
                                samples.append(full_code.strip())
                                count += 1
                    except json.JSONDecodeError:
                        continue
            else:
                print(f"   Failed to fetch HumanEval: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   Error loading HumanEval: {e}")
        
        return samples
    
    def _load_from_github_repos(self, max_samples: int) -> List[str]:
        """Load Python files from popular GitHub repositories."""
        samples = []
        
        # Popular Python files from well-known repositories
        repo_files = [
            "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
            "https://raw.githubusercontent.com/python/cpython/main/Lib/itertools.py",
            "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py",
            "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py",
            "https://raw.githubusercontent.com/pallets/flask/main/src/flask/helpers.py",
            "https://raw.githubusercontent.com/psf/requests/main/src/requests/api.py",
            "https://raw.githubusercontent.com/psf/requests/main/src/requests/utils.py",
        ]
        
        for url in repo_files:
            if len(samples) >= max_samples:
                break
                
            try:
                print(f"   Fetching {url.split('/')[-1]}...")
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    code = response.text
                    functions = self._extract_functions(code)
                    
                    # Take up to 3 functions from each file
                    for func in functions[:3]:
                        if len(samples) >= max_samples:
                            break
                        if self._is_good_code_sample(func):
                            samples.append(func)
                            
            except Exception as e:
                print(f"   Error loading {url}: {e}")
                continue
        
        return samples
    
    def _load_from_raw_sources(self, max_samples: int) -> List[str]:
        """Load Python examples from various raw sources."""
        samples = []
        
        # Some example Python snippets from various educational sources
        raw_examples = [
            # Algorithm examples
            '''
def binary_search(arr, target):
    """Binary search implementation."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
            ''',
            
            # Data structure examples
            '''
class Stack:
    """Simple stack implementation."""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item."""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        """Check if stack is empty."""
        return len(self.items) == 0
            ''',
            
            # API examples
            '''
import json
import urllib.request

def fetch_weather_data(city):
    """Fetch weather data for a city."""
    try:
        url = f"https://api.weather.com/data/{city}"
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
        return data
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None
            ''',
            
            # File processing examples
            '''
def process_csv_file(filename):
    """Process a CSV file and return processed data."""
    import csv
    results = []
    
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_row = {
                    'id': int(row.get('id', 0)),
                    'name': row.get('name', '').strip(),
                    'value': float(row.get('value', 0.0))
                }
                results.append(processed_row)
    except FileNotFoundError:
        print(f"File {filename} not found")
    except ValueError as e:
        print(f"Error processing data: {e}")
    
    return results
            ''',
            
            # Poor examples for contrast
            '''
def f(x,y,z):
    return x+y*z if z>0 else x-y
            ''',
            
            '''
def a(b):
    c=[]
    for d in b:
        if d>0:c.append(d*2)
        else:c.append(d)
    return c
            ''',
        ]
        
        # Add examples up to max_samples
        return raw_examples[:max_samples]
    
    def _get_quality_examples(self, max_samples: int) -> List[str]:
        """Get high-quality Python examples with different readability levels."""
        examples = [
            # Score 5 - Excellent
            '''
def calculate_compound_interest(principal: float, rate: float, time: float, compound_freq: int = 1) -> float:
    """
    Calculate compound interest using the standard formula.
    
    Args:
        principal: Initial amount of money
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time period in years
        compound_freq: Number of times interest is compounded per year
    
    Returns:
        Final amount after compound interest
    
    Raises:
        ValueError: If any parameter is negative
    
    Example:
        >>> calculate_compound_interest(1000, 0.05, 2, 4)
        1104.49
    """
    if principal < 0 or rate < 0 or time < 0 or compound_freq <= 0:
        raise ValueError("All parameters must be positive")
    
    amount = principal * (1 + rate / compound_freq) ** (compound_freq * time)
    return round(amount, 2)
            ''',
            
            # Score 4 - Good
            '''
def find_duplicates(numbers):
    """Find duplicate numbers in a list."""
    seen = set()
    duplicates = set()
    
    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)
            ''',
            
            # Score 3 - Moderate
            '''
def process_orders(orders):
    total = 0
    processed = []
    for order in orders:
        if order['status'] == 'pending':
            order['status'] = 'processed'
            total += order['amount']
            processed.append(order)
    return total, processed
            ''',
            
            # Score 2 - Poor
            '''
def calc(x,y,op):
    if op=='+':return x+y
    elif op=='-':return x-y
    elif op=='*':return x*y
    elif op=='/':return x/y if y!=0 else 0
            ''',
            
            # Score 1 - Very poor
            '''
def f(a,b,c,d):
    return(a+b)*(c-d)if c>d else(a-b)/(c+d)if c+d!=0 else 0
            ''',
        ]
        
        return examples[:max_samples]
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract individual functions from a code file."""
        functions = []
        lines = code.split('\n')
        current_function = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments at module level
            if not in_function and (not stripped or stripped.startswith('#')):
                continue
            
            # Check for function definition
            if stripped.startswith('def '):
                if in_function and current_function:
                    # Save previous function
                    func_code = '\n'.join(current_function).strip()
                    if len(func_code) > 50 and len(func_code) < 1500:
                        functions.append(func_code)
                
                # Start new function
                current_function = [line]
                in_function = True
                indent_level = len(line) - len(line.lstrip())
            
            elif in_function:
                line_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
                
                # If we're still in the function (indented properly) or it's an empty line
                if line_indent > indent_level or not line.strip():
                    current_function.append(line)
                else:
                    # Function ended, save it
                    func_code = '\n'.join(current_function).strip()
                    if len(func_code) > 50 and len(func_code) < 1500:
                        functions.append(func_code)
                    
                    # Check if this line starts a new function
                    if stripped.startswith('def '):
                        current_function = [line]
                        indent_level = len(line) - len(line.lstrip())
                    else:
                        in_function = False
                        current_function = []
        
        # Don't forget the last function
        if in_function and current_function:
            func_code = '\n'.join(current_function).strip()
            if len(func_code) > 50 and len(func_code) < 1500:
                functions.append(func_code)
        
        return functions
    
    def _is_good_code_sample(self, code: str) -> bool:
        """Check if a code sample is suitable for readability analysis."""
        if not code or len(code.strip()) < 30:
            return False
        
        # Must contain Python keywords
        python_keywords = ['def', 'class', 'import', 'if', 'for', 'while', 'return']
        if not any(keyword in code for keyword in python_keywords):
            return False
        
        # Should not be too long
        if len(code) > 2000:
            return False
        
        # Should not be mostly comments
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > len(lines) * 0.6:
            return False
        
        return True
    
    def _get_local_examples(self) -> List[str]:
        """Get local code examples as fallback."""
        return [
            '''
def merge_sort(arr):
    """Merge sort implementation with clear variable names."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])
    
    return merge(left_half, right_half)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
            ''',
            
            '''
def x(a,b):
    return a+b if a>b else a*b
            ''',
        ]
    
    def _label_with_openai(self, code_samples: List[str]) -> bool:
        """Label code samples using OpenAI API."""
        print(f"\n🏷️  Labeling {len(code_samples)} samples with OpenAI...")
        
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["code_snippet", "readability_score", "reason", "tokens", "lines", "source"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, code in enumerate(code_samples):
                print(f"   Progress: {i+1}/{len(code_samples)}")
                
                score, reason = self._get_openai_score(code)
                
                if score is not None:
                    writer.writerow({
                        "code_snippet": code.replace('\n', '\\n'),
                        "readability_score": score,
                        "reason": reason or "No reason provided",
                        "tokens": len(code.split()),
                        "lines": len(code.split('\n')),
                        "source": "real_dataset"
                    })
                    self.processed_count += 1
                else:
                    self.error_count += 1
                
                time.sleep(0.3)  # Rate limiting
        
        print(f"✅ Labeled {self.processed_count} samples successfully")
        return True
    
    def _label_with_heuristics(self, code_samples: List[str]) -> bool:
        """Label code samples using heuristic rules (no API needed)."""
        print(f"\n🏷️  Labeling {len(code_samples)} samples with heuristics...")
        
        with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["code_snippet", "readability_score", "reason", "tokens", "lines", "source"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for code in code_samples:
                score, reason = self._get_heuristic_score(code)
                
                writer.writerow({
                    "code_snippet": code.replace('\n', '\\n'),
                    "readability_score": score,
                    "reason": reason,
                    "tokens": len(code.split()),
                    "lines": len(code.split('\n')),
                    "source": "real_dataset_heuristic"
                })
                self.processed_count += 1
        
        print(f"✅ Labeled {self.processed_count} samples with heuristics")
        return True
    
    def _get_openai_score(self, code: str) -> Tuple[Optional[int], Optional[str]]:
        """Get readability score from OpenAI."""
        try:
            prompt = f"""
Rate this Python code's readability on a scale of 1-5:
1 = Very hard to read (unclear names, no structure)
2 = Hard to read (some issues)
3 = Moderately readable (decent but could improve)
4 = Good readability (clear and well-structured)
5 = Excellent readability (perfect naming, docs, structure)

Code:
```python
{code[:800]}
```

Respond ONLY in format:
Score: [1-5]
Reason: [brief explanation]
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a code readability expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response
            score_match = re.search(r'Score:\s*([1-5])', content)
            reason_match = re.search(r'Reason:\s*(.+)', content, re.IGNORECASE)
            
            if score_match and reason_match:
                return int(score_match.group(1)), reason_match.group(1).strip()
            
        except Exception as e:
            print(f"   OpenAI error: {e}")
        
        return None, None
    
    def _get_heuristic_score(self, code: str) -> Tuple[int, str]:
        """Score code readability using heuristic rules."""
        score = 3  # Start with moderate
        reasons = []
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Check documentation
        if '"""' in code or "'''" in code:
            score += 1
            reasons.append("has docstrings")
        
        # Check function/variable naming
        if re.search(r'def\s+[a-z][a-z0-9_]*\s*\(', code):
            score += 0.5
            reasons.append("good function naming")
        elif re.search(r'def\s+[a-zA-Z]\s*\(', code):
            score -= 1
            reasons.append("poor function naming")
        
        # Check line length
        long_lines = sum(1 for line in lines if len(line) > 80)
        if long_lines > len(non_empty_lines) * 0.3:
            score -= 1
            reasons.append("many long lines")
        
        # Check comments
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / max(len(non_empty_lines), 1)
        if comment_ratio > 0.1:
            score += 0.5
            reasons.append("good comments")
        
        # Clamp score to 1-5 range
        score = max(1, min(5, int(score)))
        
        reason = "; ".join(reasons) if reasons else "moderate readability"
        
        return score, reason

def main():
    """Main function to run the dataset loading."""
    print("=" * 60)
    print("🚀 COGNITO DATASET LOADER - DIRECT ACCESS")
    print("=" * 60)
    
    # Check for API key (optional)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key found - will use GPT for labeling")
    else:
        print("⚠️  No OpenAI API key found - will use heuristic labeling")
    
    # Get user preferences
    test_mode = input("Run in test mode (10 samples)? [y/N]: ").lower().startswith('y')
    
    if not test_mode:
        num_samples = input("How many samples to collect? [20]: ")
        try:
            num_samples = int(num_samples) if num_samples else 20
        except ValueError:
            num_samples = 20
    else:
        num_samples = 10
    
    # Create loader and run
    loader = DatasetLoader(api_key)
    success = loader.load_real_datasets(num_samples, test_mode)
    
    if success:
        print(f"\n🎉 Dataset created successfully!")
        print(f"📁 Output: {loader.output_file}")
        print(f"📊 Processed: {loader.processed_count} samples")
        if loader.error_count > 0:
            print(f"⚠️  Errors: {loader.error_count}")
        
        # Show sample of what was created
        try:
            import pandas as pd
            df = pd.read_csv(loader.output_file)
            print(f"\n📊 Dataset preview:")
            print(f"   Score distribution: {df['readability_score'].value_counts().sort_index().to_dict()}")
            print(f"   Average tokens: {df['tokens'].mean():.1f}")
            print(f"   Average lines: {df['lines'].mean():.1f}")
            print(f"   Total samples: {len(df)}")
        except Exception as e:
            print(f"   Could not show preview: {e}")
            
        print(f"\n🔬 Next steps:")
        print(f"1. Validate dataset: python src/data/dataset_validator.py")
        print(f"2. Train model: python src/models/fine_tune_readability.py")
    else:
        print(f"\n❌ Dataset creation failed")

if __name__ == "__main__":
    main()