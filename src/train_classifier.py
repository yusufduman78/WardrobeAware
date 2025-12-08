import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src import fclip_config as config
from utils import fclip_utils as utils

def train_classifier():
    print("Loading data...")
    # Load embeddings
    if not config.IMAGE_EMBEDDINGS_PATH.exists():
        print(f"Error: Embeddings not found at {config.IMAGE_EMBEDDINGS_PATH}")
        return

    embeddings = np.load(config.IMAGE_EMBEDDINGS_PATH)
    
    # Load item IDs
    item_ids = utils.load_json(config.ITEM_IDS_PATH)
    
    # Load metadata
    metadata = utils.load_metadata()
    
    print(f"Loaded {len(embeddings)} embeddings and {len(item_ids)} item IDs.")
    
    # Prepare X and y
    X = []
    y = []
    
    valid_indices = []
    
    print("Preparing dataset...")
    for i, item_id in enumerate(item_ids):
        if item_id in metadata:
            # Use semantic_category instead of category_id
            semantic_category = metadata[item_id].get('semantic_category')
            if semantic_category:
                X.append(embeddings[i])
                y.append(semantic_category)
                valid_indices.append(i)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Number of unique categories: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    print("Training Logistic Regression Classifier...")
    # Using Logistic Regression as it's fast and effective for high-dim features
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Save Model
    model_path = config.MODELS_DIR / "category_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Model saved to {model_path}")
    
    # Save Label Encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    label_encoder_path = config.MODELS_DIR / "category_label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"Label encoder saved to {label_encoder_path}")
    print(f"Categories: {list(label_encoder.classes_)}")

if __name__ == "__main__":
    train_classifier()
