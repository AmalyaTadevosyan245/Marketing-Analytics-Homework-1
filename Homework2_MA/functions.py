import pandas as pd

def save_results(df, filename="results.csv"):
    """Save combined experiment results to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
