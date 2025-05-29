import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def validate_dataset(csv_path="src/data/readability_dataset.csv"):
    """
    Validate the quality of the generated readability dataset.
    
    Args:
        csv_path: Path to the CSV dataset file
    """
    if not os.path.exists(csv_path):
        print(f"Dataset file not found: {csv_path}")
        return False
    
    print("=" * 50)
    
    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Basic statistics
        print(f"\nBASIC STATISTICS:")
        print(f"   • Total samples: {len(df)}")
        print(f"   • Columns: {list(df.columns)}")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        
        # Score distribution
        print(f"\nREADABILITY SCORE DISTRIBUTION:")
        score_counts = df['readability_score'].value_counts().sort_index()
        for score, count in score_counts.items():
            percentage = count / len(df) * 100
            bar = "█" * int(percentage / 2)
            print(f"   Score {score}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Code length statistics
        if 'tokens' in df.columns:
            print(f"\nCODE LENGTH STATISTICS:")
            print(f"   • Average tokens: {df['tokens'].mean():.1f}")
            print(f"   • Average lines: {df['lines'].mean():.1f}")
            print(f"   • Token range: {df['tokens'].min()} - {df['tokens'].max()}")
            print(f"   • Line range: {df['lines'].min()} - {df['lines'].max()}")
        
        # Sample reasons for each score
        print(f"\nSAMPLE REASONS BY SCORE:")
        for score in sorted(df['readability_score'].unique()):
            sample_reasons = df[df['readability_score'] == score]['reason'].head(3).tolist()
            print(f"   Score {score}:")
            for reason in sample_reasons:
                print(f"     • {reason[:80]}...")
        
        # Check for duplicate code
        duplicates = df['code_snippet'].duplicated().sum()
        print(f"   • Duplicate code samples: {duplicates}")
        
        # Check for very short code
        if 'tokens' in df.columns:
            short_code = (df['tokens'] < 10).sum()
            print(f"   • Very short code (<10 tokens): {short_code}")
        
        # Check for missing reasons
        missing_reasons = df['reason'].isnull().sum()
        print(f"   • Missing reasons: {missing_reasons}")
        
        # Check score distribution balance
        min_score_count = score_counts.min()
        max_score_count = score_counts.max()
        imbalance_ratio = max_score_count / min_score_count
        print(f"   • Class imbalance ratio: {imbalance_ratio:.2f} (ideally < 3.0)")
        
        # Overall quality assessment
        print(f"\nOVERALL QUALITY ASSESSMENT:")
        quality_score = 100
        
        if duplicates > len(df) * 0.05:  # More than 5% duplicates
            quality_score -= 20
            print("   ⚠️  High duplicate rate")
        
        if missing_reasons > len(df) * 0.1:  # More than 10% missing reasons
            quality_score -= 15
            print("   ⚠️  Many missing reasons")
        
        if imbalance_ratio > 5.0:  # Very imbalanced classes
            quality_score -= 15
            print("   ⚠️  Highly imbalanced classes")
        
        if len(df) < 500:  # Too few samples
            quality_score -= 20
            print("   ⚠️  Dataset too small for reliable training")
        
        if quality_score >= 90:
            print(f"   ✅ Excellent quality ({quality_score}/100)")
        elif quality_score >= 70:
            print(f"   ✅ Good quality ({quality_score}/100)")
        elif quality_score >= 50:
            print(f"   ⚠️  Fair quality ({quality_score}/100)")
        else:
            print(f"   ❌\ Poor quality ({quality_score}/100)")
        
        # Recommendations
        print(f"\Recommentations:")
        if len(df) < 1000:
            print("   • Consider generating more samples (target: 1000+)")
        if imbalance_ratio > 3.0:
            print("   • Consider balancing classes by generating more low-frequency scores")
        if duplicates > 0:
            print("   • Remove duplicate samples before training")
        
        print(f"\nValidation complete! Dataset is {'ready' if quality_score >= 70 else 'needs improvement'} for training.")
        return quality_score >= 70
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False

def show_sample_data(csv_path="src/data/readability_dataset.csv", n=5):
    """Show sample data from the dataset."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\nSAMPLE DATA (first {n} rows):")
        print("-" * 100)
        
        for i, row in df.head(n).iterrows():
            code_preview = row['code_snippet'].replace('\\n', '\n')[:200] + "..." if len(row['code_snippet']) > 200 else row['code_snippet'].replace('\\n', '\n')
            print(f"Sample {i+1}:")
            print(f"  Score: {row['readability_score']}")
            print(f"  Reason: {row['reason']}")
            print(f"  Code preview:")
            print(f"    {code_preview}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error showing sample data: {e}")

if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "src/data/readability_dataset.csv"
    print(f"Validating dataset at: {csv_path}")
    print("=" * 50)
    
    # Validate dataset
    is_valid = validate_dataset(csv_path)
    
    # Show samples if requested
    show_samples = input("\nShow sample data? [y/N]: ").lower().startswith('y')
    if show_samples:
        show_sample_data(csv_path)
    
    if is_valid:
        print(f"\nNext step: Run python src/models/fine_tune_readability.py")
    else:
        print(f"\nConsider regenerating dataset with improvements")