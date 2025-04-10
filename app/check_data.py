import pandas as pd
import os

def check_dataset():
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # Read the datasets
    print("Reading True.csv...")
    true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))
    print("\nSample of True.csv:")
    print(true_df.head(2))
    print(f"\nTotal true news articles: {len(true_df)}")
    
    print("\nReading Fake.csv...")
    fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
    print("\nSample of Fake.csv:")
    print(fake_df.head(2))
    print(f"\nTotal fake news articles: {len(fake_df)}")
    
    # Read the combined dataset
    print("\nReading combined dataset (news.csv)...")
    combined_df = pd.read_csv(os.path.join(data_dir, 'news.csv'))
    print("\nSample of combined dataset:")
    print(combined_df.head(2))
    print(f"\nTotal articles in combined dataset: {len(combined_df)}")
    print(f"Real news in combined dataset: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Fake news in combined dataset: {len(combined_df[combined_df['label'] == 0])}")

if __name__ == '__main__':
    check_dataset() 