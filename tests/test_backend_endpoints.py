import requests
import json
import random

BASE_URL = "http://localhost:8000"

def test_feed():
    print("\n--- Testing /feed ---")
    try:
        response = requests.get(f"{BASE_URL}/feed?limit=5")
        if response.status_code == 200:
            items = response.json()
            print(f"Success: Got {len(items)} items")
            return items[0]['id'] if items else None
        else:
            print(f"Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_outfit_complete(item_id):
    print(f"\n--- Testing /outfit/complete with item {item_id} ---")
    if not item_id:
        print("Skipping: No item ID available")
        return

    try:
        payload = {"item_id": item_id}
        response = requests.post(f"{BASE_URL}/outfit/complete", json=payload)
        if response.status_code == 200:
            recs = response.json()
            print(f"Success: Got {len(recs)} recommendations")
            for rec in recs:
                print(f"  - {rec['name']} ({rec.get('category', 'unknown')}) - Score: {rec['score']:.2f}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_combinations(user_id="test_user"):
    print(f"\n--- Testing /outfit/combinations for user {user_id} ---")
    try:
        response = requests.get(f"{BASE_URL}/outfit/combinations?user_id={user_id}")
        if response.status_code == 200:
            combos = response.json()
            print(f"Success: Got {len(combos)} combinations")
            for combo in combos:
                items = combo['items']
                print(f"  - Combo: {items[0]['name']} + {items[1]['name']} - Score: {combo['score']:.2f}")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_register():
    print("\n--- Testing /auth/register ---")
    try:
        # Random username to avoid conflict
        username = f"test_user_{random.randint(1000, 9999)}"
        payload = {
            "username": username,
            "password": "password123"
            # email is omitted to test optionality
        }
        response = requests.post(f"{BASE_URL}/auth/register", json=payload)
        if response.status_code == 200:
            print(f"Success: Registered {username}")
            print(response.json())
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting Backend Tests...")
    item_id = test_feed()
    if item_id:
        test_outfit_complete(item_id)
    
    # Ensure we have some likes for test_user (mocking if needed, but assuming DB has some from previous swipes)
    # If not, we might get empty list, which is valid but not fully verifying.
    # Let's try to swipe some items first to ensure data exists
    if item_id:
        print("\n--- Seeding Likes ---")
        requests.post(f"{BASE_URL}/swipe", json={"user_id": "test_user", "item_id": item_id, "action": "like"})
        # We need at least 2 items for combinations. Let's get another one.
        # But for now let's just run the test.
        
    test_combinations()
    test_register()
