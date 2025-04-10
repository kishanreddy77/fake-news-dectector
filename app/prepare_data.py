import pandas as pd
import os

def prepare_dataset():
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Read the datasets
    print("Reading True.csv...")
    true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))
    true_df['label'] = 1  # 1 for real news
    
    print("Reading Fake.csv...")
    fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
    fake_df['label'] = 0  # 0 for fake news
    
    # Combine the datasets
    print("Combining datasets...")
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Ensure we have the required columns
    if 'text' not in combined_df.columns and 'title' in combined_df.columns and 'text' in combined_df.columns:
        # Combine title and text if they exist separately
        combined_df['text'] = combined_df['title'] + ' ' + combined_df['text']
    
    # Keep only necessary columns
    combined_df = combined_df[['text', 'label']]
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the combined dataset
    print("Saving combined dataset...")
    output_path = os.path.join(data_dir, 'news.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path}!")
    print(f"Total samples: {len(combined_df)}")
    print(f"Real news: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Fake news: {len(combined_df[combined_df['label'] == 0])}")

if __name__ == '__main__':
    prepare_dataset() 