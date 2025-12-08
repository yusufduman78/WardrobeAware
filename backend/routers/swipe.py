from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from backend.database import get_db_connection
import sqlite3

router = APIRouter(prefix="/swipe", tags=["swipe"])

class SwipeRequest(BaseModel):
    user_id: str
    item_id: str
    action: str  # 'like', 'dislike', 'superlike', 'save'

@router.post("")
async def swipe_item(request: SwipeRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Log interaction
        print(f"DEBUG: User {request.user_id} performed {request.action} on {request.item_id}")
        cursor.execute(
            "INSERT INTO interactions (user_id, item_id, action) VALUES (?, ?, ?)",
            (request.user_id, request.item_id, request.action)
        )
        
        # Update User Taste Vector
        # 1. Get Item Embedding
        # We need to load embeddings. For efficiency, we should load them once globally or use a database.
        # For MVP, we'll load on demand (slow) or assume we have a helper.
        # Let's use a simplified approach: We assume we can get the embedding from a global store.
        
        # TODO: In a real app, use a vector DB. Here we mock the vector update logic.
        # If we had the vector:
        # current_vector = get_user_vector(request.user_id)
        # item_vector = get_item_vector(request.item_id)
        # alpha = 0.1
        # weight = 1.0 if request.action == 'like' else (2.0 if request.action == 'superlike' else -0.5)
        # new_vector = current_vector * (1 - alpha) + item_vector * weight * alpha
        # save_user_vector(request.user_id, new_vector)
        
        print(f"DEBUG: Would update user vector for {request.user_id} based on {request.action}")
        
        conn.commit()
        return {"status": "success", "message": f"Recorded {request.action} for {request.item_id}"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
