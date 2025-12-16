import json
from pathlib import Path
import sys

# Define path to metadata
# Based on config.py: BASE_DIR / "data" / "polyvore_outfits" / "polyvore_item_metadata.json"
# We'll calculate it relative to current script
current_dir = Path.cwd()
metadata_path = current_dir / "data" / "polyvore_outfits" / "polyvore_item_metadata.json"

if not metadata_path.exists():
    print(f"Error: Metadata file not found at {metadata_path}")
    sys.exit(1)

try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    categories = set()
    for item in data.values():
        cat = item.get('semantic_category')
        if cat:
            categories.add(cat)
            
    sorted_cats = sorted(list(categories))
    print(f"Found {len(sorted_cats)} unique categories:")
    for cat in sorted_cats:
        print(f"- {cat}")
        
except Exception as e:
    print(f"Error reading metadata: {e}")
