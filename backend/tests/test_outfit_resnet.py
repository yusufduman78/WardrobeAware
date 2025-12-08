from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.main import app

client = TestClient(app)

def test_outfit_complete_resnet():
    # Use a known item ID (from debug_overlap.py output)
    item_id = "211990161" 
    
    print(f"Testing /outfit/complete with item_id={item_id}")
    
    response = client.post("/outfit/complete", json={"item_id": item_id})
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        assert response.status_code == 200
        
    data = response.json()
    print("Response received.")
    
    assert "recommendations" in data
    recommendations = data["recommendations"]
    
    print(f"Received {len(recommendations)} recommendations.")
    
    if len(recommendations) > 0:
        first_rec = recommendations[0]
        print("First recommendation sample:")
        print(first_rec)
        
        assert "id" in first_rec
        assert "image_url" in first_rec
        assert "match_score" in first_rec
        assert "category" in first_rec
        
        # Check if score is integer percentage
        assert isinstance(first_rec["match_score"], int)
        assert 0 <= first_rec["match_score"] <= 100
        
        print("Test Passed!")
    else:
        print("Warning: No recommendations returned (might be due to random sampling or missing data)")

def test_create_outfit_from_anchor():
    item_id = "211990161" # Known ID
    print(f"Testing /outfit/create_from_anchor with item_id={item_id}")
    
    response = client.post("/outfit/create_from_anchor", json={"anchor_item_id": item_id})
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        
    assert response.status_code == 200
    data = response.json()
    
    assert "items" in data
    assert "match_score" in data
    assert len(data["items"]) > 1
    
    # Check if anchor is in items
    anchor_in_items = any(item['id'] == item_id for item in data['items'])
    assert anchor_in_items
    
    # Check if items have order
    assert "order" in data["items"][0]
    print("test_create_outfit_from_anchor Passed!")

if __name__ == "__main__":
    import sys
    # Redirect stdout/stderr to file
    with open("test_output.log", "w") as f:
        sys.stdout = f
        sys.stderr = f
        try:
            test_outfit_complete_resnet()
            test_create_outfit_from_anchor()
            print("All tests passed!")
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
