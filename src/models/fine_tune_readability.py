# src/models/fine_tune_readability.py - Fix for small datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re

class ReadabilityModelTrainer:
    """Train a readability classification model using traditional ML - FIXED for small datasets."""
    
    def __init__(self, dataset_path="src/data/readability_dataset.csv"):
        """Initialize the trainer."""
        self.dataset_path = dataset_path
        self.model = None
        self.vectorizer = None
        self.model_path = "src/models/readability_model.pkl"
        self.vectorizer_path = "src/models/readability_vectorizer.pkl"
        
        # Ensure models directory exists
        os.makedirs("src/models", exist_ok=True)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        print("Loading dataset...")
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(df)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
        
        # Display dataset info
        print(f"Score distribution: {df['readability_score'].value_counts().sort_index().to_dict()}")
        
        # FIXED: Handle small datasets by augmenting data
        if len(df) < 20:
            print("⚠️  Small dataset detected. Augmenting data...")
            df = self._augment_small_dataset(df)
            print(f"Augmented to {len(df)} samples")
            print(f"New score distribution: {df['readability_score'].value_counts().sort_index().to_dict()}")
        
        # Extract features from code
        print("Extracting features...")
        features = []
        
        for code in df['code_snippet']:
            # Restore newlines
            code = code.replace('\\n', '\n')
            
            # Extract readability features
            feature_dict = self._extract_readability_features(code)
            features.append(feature_dict)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Combine text features with numerical features
        X_text = df['code_snippet'].str.replace('\\n', ' ')  # Text for TF-IDF
        X_numerical = feature_df
        y = df['readability_score']
        
        return X_text, X_numerical, y
    
    def _augment_small_dataset(self, df):
        """Augment small dataset with synthetic variations."""
        augmented_data = []
        
        for _, row in df.iterrows():
            augmented_data.append(row.to_dict())
            
            # Create variations for underrepresented classes
            score = row['readability_score']
            code = row['code_snippet']
            
            # Add variations only for classes with < 3 samples
            if (df['readability_score'] == score).sum() < 3:
                # Variation 1: Add comments
                code_with_comments = code.replace('def ', '# Function definition\ndef ')
                augmented_data.append({
                    **row.to_dict(),
                    'code_snippet': code_with_comments,
                    'reason': row['reason'] + ' (with comments)',
                    'source': 'augmented_comments'
                })
                
                # Variation 2: Change variable names (for poor readability)
                if score <= 2:
                    code_bad_vars = code.replace('number', 'x').replace('result', 'r').replace('value', 'v')
                    augmented_data.append({
                        **row.to_dict(),
                        'code_snippet': code_bad_vars,
                        'reason': row['reason'] + ' (poor variable names)',
                        'source': 'augmented_bad_vars'
                    })
                
                # Variation 3: Format variations
                if score >= 4:
                    # Well-formatted version
                    code_formatted = code.replace(';', ';\n    ')
                    augmented_data.append({
                        **row.to_dict(),
                        'code_snippet': code_formatted,
                        'reason': row['reason'] + ' (well-formatted)',
                        'source': 'augmented_formatted'
                    })
        
        # Add some generic examples to balance classes
        generic_examples = [
            {
                'code_snippet': 'def a(b,c):return b+c',
                'readability_score': 1,
                'reason': 'Very poor formatting and naming',
                'tokens': 7,
                'lines': 1,
                'source': 'augmented_generic'
            },
            {
                'code_snippet': 'def add_numbers(first, second):\n    """Add two numbers together."""\n    return first + second',
                'readability_score': 5,
                'reason': 'Excellent naming and documentation',
                'tokens': 15,
                'lines': 3,
                'source': 'augmented_generic'
            },
            {
                'code_snippet': 'def calc(x,y):\n  return x*y',
                'readability_score': 2,
                'reason': 'Poor naming but basic structure',
                'tokens': 8,
                'lines': 2,
                'source': 'augmented_generic'
            },
            {
                'code_snippet': 'def multiply(num1, num2):\n    result = num1 * num2\n    return result',
                'readability_score': 4,
                'reason': 'Good naming and structure',
                'tokens': 12,
                'lines': 3,
                'source': 'augmented_generic'
            },
            {
                'code_snippet': 'def process_data(data):\n    # TODO: implement\n    pass',
                'readability_score': 3,
                'reason': 'Moderate readability with placeholder',
                'tokens': 9,
                'lines': 3,
                'source': 'augmented_generic'
            }
        ]
        
        augmented_data.extend(generic_examples)
        
        return pd.DataFrame(augmented_data)
    
    def train_model(self):
        """Train the readability classification model."""
        # Load data
        X_text, X_numerical, y = self.load_and_preprocess_data()
        
        if X_text is None:
            print("Failed to load data")
            return False
        
        print("Training model...")
        
        # Create TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(
            max_features=100,  # Reduced for small dataset
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Transform text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X_text).toarray()
        
        # Combine TF-IDF with numerical features
        X_combined = np.hstack([X_tfidf, X_numerical.values])
        
        # FIXED: Handle small datasets without stratification
        if len(y.unique()) < 3 or len(y) < 10:
            print("⚠️  Very small dataset - using simple train/test split")
            # Simple split without stratification
            test_size = min(0.3, 2/len(y))  # At least 2 samples for test, max 30%
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=test_size, random_state=42
            )
        else:
            # Check if stratification is possible
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                print("⚠️  Cannot stratify - some classes have only 1 sample")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y, test_size=0.2, random_state=42
                )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced for small dataset
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            min_samples_split=2,  # Allow small splits
            min_samples_leaf=1    # Allow single sample leaves
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Only show classification report if we have test samples
        if len(y_test) > 0:
            print("\nClassification Report:")
            try:
                print(classification_report(y_test, y_pred, zero_division=0))
            except:
                print("Could not generate classification report (likely due to small test set)")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = ([f'tfidf_{i}' for i in range(X_tfidf.shape[1])] + 
                            list(X_numerical.columns))
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Most Important Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Save model and vectorizer
        self._save_model()
        
        return accuracy
    
    def _extract_readability_features(self, code):
        """Extract numerical features that indicate code readability."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        features = {}
        
        # Basic metrics
        features['total_lines'] = len(lines)
        features['non_empty_lines'] = len(non_empty_lines)
        features['total_chars'] = len(code)
        features['avg_line_length'] = np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        
        # Documentation features
        features['has_docstring'] = 1 if ('"""' in code or "'''" in code) else 0
        features['comment_ratio'] = sum(1 for line in lines if line.strip().startswith('#')) / max(len(non_empty_lines), 1)
        
        # Naming features
        features['good_function_names'] = len(re.findall(r'def\s+[a-z][a-z0-9_]*\s*\(', code))
        features['poor_function_names'] = len(re.findall(r'def\s+[a-zA-Z]\s*\(', code)) - features['good_function_names']
        features['total_functions'] = len(re.findall(r'def\s+\w+', code))
        
        # Complexity features
        features['nested_loops'] = len(re.findall(r'for.*:\s*\n\s+.*for.*:', code, re.MULTILINE))
        features['if_statements'] = len(re.findall(r'\bif\b', code))
        features['for_loops'] = len(re.findall(r'\bfor\b', code))
        features['while_loops'] = len(re.findall(r'\bwhile\b', code))
        
        # Style features
        features['long_lines'] = sum(1 for line in lines if len(line) > 80)
        features['very_long_lines'] = sum(1 for line in lines if len(line) > 120)
        
        # Keyword density
        python_keywords = ['def', 'class', 'import', 'if', 'else', 'for', 'while', 'try', 'except', 'return']
        features['keyword_density'] = sum(code.count(kw) for kw in python_keywords) / max(len(code.split()), 1)
        
        return features
    
    def _save_model(self):
        """Save the trained model and vectorizer."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"\n✅ Model saved to {self.model_path}")
            print(f"✅ Vectorizer saved to {self.vectorizer_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
                return False
                
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_readability(self, code):
        """Predict readability score for a code snippet."""
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                return None, "Model not available"
        
        # Extract features
        features = self._extract_readability_features(code)
        X_numerical = pd.DataFrame([features])
        
        # Transform text
        X_tfidf = self.vectorizer.transform([code.replace('\n', ' ')]).toarray()
        
        # Combine features
        X_combined = np.hstack([X_tfidf, X_numerical.values])
        
        # Predict
        prediction = self.model.predict(X_combined)[0]
        probability = self.model.predict_proba(X_combined)[0]
        
        # Generate explanation
        confidence = max(probability)
        
        if prediction <= 2:
            explanation = f"Poor readability (score {prediction}) - consider improving naming and structure"
        elif prediction <= 3:
            explanation = f"Moderate readability (score {prediction}) - some improvements possible"
        elif prediction <= 4:
            explanation = f"Good readability (score {prediction}) - well-structured code"
        else:
            explanation = f"Excellent readability (score {prediction}) - very clear and well-documented"
        
        return prediction, f"{explanation} (confidence: {confidence:.2f})"

def main():
    """Main training function."""
    print("=" * 60)
    print("🤖 COGNITO READABILITY MODEL TRAINER - FIXED")
    print("=" * 60)
    
    trainer = ReadabilityModelTrainer()
    
    # Check if dataset exists
    if not os.path.exists(trainer.dataset_path):
        print(f"❌ Dataset not found at {trainer.dataset_path}")
        print("Please run: python src/data/dataset_loader.py first")
        return
    
    # Train model
    accuracy = trainer.train_model()
    
    if accuracy and accuracy > 0:
        print(f"\n🎉 Training completed!")
        print(f"📊 Final accuracy: {accuracy*100:.1f}%")
        
        # Test the model
        print("\n🧪 Testing model...")
        
        test_cases = [
            "def f(x): return x*2",
            "def calculate_double(number): return number * 2",
            '''
def calculate_fibonacci(n):
    """Calculate Fibonacci number recursively."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
            '''
        ]
        
        for i, code in enumerate(test_cases, 1):
            try:
                score, explanation = trainer.predict_readability(code)
                print(f"\nTest {i}: {code.split('def')[1].split(':')[0] if 'def' in code else code[:30]}...")
                print(f"  Result: {explanation}")
            except Exception as e:
                print(f"  Error testing: {e}")
        
        print(f"\n🔬 Next steps:")
        print(f"1. Model is now available for use in readability analysis")
        print(f"2. Test with: cognito --file test_programs/sum.py")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()