import pandas as pd
import numpy as np
import pickle
import os
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define file paths
dataset_files = {
    'gossipcop_fake': os.path.join(os.path.dirname(__file__), 'gossipcop_fake.csv'),
    'gossipcop_real': os.path.join(os.path.dirname(__file__), 'gossipcop_real.csv'),
    'politifact_fake': os.path.join(os.path.dirname(__file__), 'politifact_fake.csv'),
    'politifact_real': os.path.join(os.path.dirname(__file__), 'politifact_real.csv')
}

MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'fake_news_model.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to prepare data for prediction
def prepare_data(df):
    """
    Prepare data for prediction by handling different column names and preprocessing text
    """
    print(f"Available columns in DataFrame: {df.columns.tolist()}")

    # Clean column names - convert to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()

    # Define possible column names
    title_cols = ['title', 'headline', 'header']
    text_cols = ['text', 'content', 'body', 'article']

    # Find the actual column names in the DataFrame
    title_col = next((col for col in title_cols if col in df.columns), None)
    text_col = next((col for col in text_cols if col in df.columns), None)

    # Create content column based on available data
    if title_col and text_col:
        df['content'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    elif title_col:
        df['content'] = df[title_col].fillna('')
    elif text_col:
        df['content'] = df[text_col].fillna('')
    else:
        raise ValueError(f"No valid text columns found. Available columns: {df.columns.tolist()}")

    df['processed_content'] = df['content'].apply(preprocess_text)
    return df

# Load and prepare dataset
def load_and_prepare_data():
    print("\nLoading and preparing dataset...")
    df_list = []
    missing_files = []

    for label, file_path in dataset_files.items():
        if os.path.exists(file_path):
            try:
                # Read CSV with proper encoding and handling
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                print(f"\nProcessing {file_path}")
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Number of rows: {len(df)}")

                # Clean column names - remove trailing spaces and backslashes
                df.columns = df.columns.str.strip().str.rstrip('\\').str.strip()
                print(f"Cleaned columns: {df.columns.tolist()}")

                # Create new dataframe with required columns
                new_df = pd.DataFrame()

                # Clean and process title
                if 'title' in df.columns:
                    new_df['title'] = df['title'].fillna('')
                else:
                    print(f"Warning: 'title' column not found in {file_path}")
                    continue

                # Add news_url as additional context
                if 'news_url' in df.columns:
                    new_df['text'] = df['news_url'].fillna('').apply(lambda x: str(x).strip())
                else:
                    new_df['text'] = ''

                # Set label based on filename
                new_df['label'] = 1 if 'fake' in label else 0

                # Clean and combine title and URL for more context
                new_df['title'] = new_df['title'].apply(lambda x: str(x).strip())
                new_df['text'] = new_df['text'].apply(lambda x: str(x).strip())
                new_df['content'] = new_df.apply(
                    lambda row: f"{row['title']} {row['text']}" if row['text'] else row['title'],
                    axis=1
                )

                # Remove any rows with empty content
                new_df = new_df[new_df['content'].str.len() > 0]

                if len(new_df) > 0:
                    print(f"Successfully processed {len(new_df)} articles from {file_path}")
                    print("Sample article:")
                    print(f"Title: {new_df['title'].iloc[0]}")
                    print(f"URL: {new_df['text'].iloc[0]}")
                    df_list.append(new_df)
                else:
                    print(f"Warning: No valid articles found in {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        else:
            missing_files.append(file_path)
            print(f"Warning: File not found - {file_path}")

    if missing_files:
        print("\nWarning: The following files were not found:", missing_files)

    if not df_list:
        raise ValueError("No valid data files found to process.")

    print("\nCombining all datasets...")
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total number of articles: {len(data)}")

    print("\nPreprocessing text...")
    data['processed_content'] = data['content'].apply(preprocess_text)

    # Remove any rows where preprocessing resulted in empty content
    data = data[data['processed_content'].str.len() > 0]
    print(f"Final number of articles after cleaning: {len(data)}")

    if len(data) == 0:
        raise ValueError("No valid text data found after preprocessing.")

    # Verify data distribution
    fake_count = sum(data['label'] == 1)
    real_count = sum(data['label'] == 0)
    print(f"\nData distribution:")
    print(f"Fake news articles: {fake_count}")
    print(f"Real news articles: {real_count}")

    return data[['processed_content', 'label']]

# Train model
def train_model():
    print("\n=== Starting Model Training ===")
    try:
        print("\n1. Loading and preparing data...")
        data = load_and_prepare_data()
        print(f"Successfully loaded {len(data)} samples")

        print("\n2. Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_content'], data['label'], test_size=0.2, random_state=42)
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        print("\n3. Creating and fitting vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        print("Vectorization completed")

        print("\n4. Training logistic regression model...")
        model = LogisticRegression(max_iter=1000, n_jobs=-1)  # Increased iterations and parallel processing
        model.fit(X_train_vectorized, y_train)
        print("Model training completed")

        print("\n5. Evaluating model performance...")
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\n6. Saving model and vectorizer...")
        save_model(model, vectorizer)

        print("\n=== Model Training Completed Successfully ===")
        return True
    except Exception as e:
        print(f"\n!!! Training Failed !!!")
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("1. Data files exist and are readable")
        print("2. Data files contain required columns")
        print("3. Sufficient disk space for model saving")
        return False

# Save model and vectorizer
def save_model(model, vectorizer):
    try:
        with open(MODEL_FILE, 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(VECTORIZER_FILE, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        print("Model and vectorizer saved successfully.")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Load model and vectorizer
def load_model():
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            with open(MODEL_FILE, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(VECTORIZER_FILE, 'rb') as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)
            return model, vectorizer
        else:
            print("Model files not found. Train the model first.")
            return None, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

if __name__ == '__main__':
    train_model()
