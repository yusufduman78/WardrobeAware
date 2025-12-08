from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database
from ..inference_service import InferenceService # Yeni yapay zeka servisi
from .auth import get_current_user
from sqlalchemy.sql.expression import func
import random

router = APIRouter(
    prefix="/feed",
    tags=["feed"]
)

@router.get("/", response_model=List[schemas.ItemOut])
def get_feed(
    limit: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # 1. Define split
    num_recommendations = int(limit * 0.6)
    num_exploration = int(limit * 0.2)
    num_reverse = limit - num_recommendations - num_exploration
    
    # 2. Get Personalized Recommendations (High Scores)
    recommendations = InferenceService().get_recommendations(current_user.id, limit=num_recommendations)
    
    for item in recommendations:
        item.image_url = item.poster_path
        item.is_recommendation = True
        
        # Classify Match
        score = getattr(item, 'score', 0.0)
        item.match_score = score
        if score > 0.75: # Threshold for perfect match
            item.match_type = "perfect"
        else:
            item.match_type = "none"

    # 3. Get Reverse Matches (Low Scores - "Definitely not your taste")
    # We need a way to get low scores. For now, let's pick random items and label them if we can't get low scores easily.
    # Ideally InferenceService should support 'ascending' sort.
    # For now, let's just use exploration items and randomly assign "reverse" for fun/demo, 
    # OR better: fetch random items and if we had scores we'd check. 
    # Since we don't have easy access to low scores without changing InferenceService API significantly,
    # let's simulate "Reverse Match" with random items for now, but label them explicitly.
    
    rec_ids = [item.id for item in recommendations]
    swiped_subquery = db.query(models.Swipe.item_id).filter(models.Swipe.user_id == current_user.id)
    
    exploration_items = db.query(models.Item).filter(
        models.Item.id.notin_(rec_ids),
        models.Item.id.notin_(swiped_subquery)
    ).order_by(func.random()).limit(num_exploration + num_reverse).all()
    
    for i, item in enumerate(exploration_items):
        item.image_url = item.poster_path
        item.is_recommendation = False
        item.match_score = 0.1 # Low score simulation
        
        if i < num_reverse:
             item.match_type = "reverse"
        else:
             item.match_type = "none"
             
        recommendations.append(item)
        
    # 4. If we still don't have enough (e.g. model returned 0), fill with popular
    if len(recommendations) < limit:
        needed = limit - len(recommendations)
        rec_ids = [item.id for item in recommendations]
        
        popular_items = db.query(models.Item).filter(
            models.Item.id.notin_(rec_ids),
            models.Item.id.notin_(swiped_subquery)
        ).order_by(models.Item.popularity.desc()).limit(needed).all()
        
        for item in popular_items:
            item.image_url = item.poster_path
            item.is_recommendation = False
            item.match_type = "none"
            recommendations.append(item)
            
    # Shuffle the final list to mix them? Or keep recommendations first?
    # Keeping recommendations first is usually better for engagement, but mixing feels more organic.
    # Let's shuffle to hide the "seam" between algos.
    random.shuffle(recommendations)
        
    return recommendations

@router.get("/match", response_model=schemas.ItemOut)
def get_match(
    match_type: str = "perfect", # "perfect" or "reverse"
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # 1. Get User Swipes to exclude
    swiped_subquery = db.query(models.Swipe.item_id).filter(models.Swipe.user_id == current_user.id)
    
    item = None
    
    if match_type == "perfect":
        # Get the absolute BEST recommendation (limit=1)
        recommendations = InferenceService().get_recommendations(current_user.id, limit=1)
        if recommendations:
            item = recommendations[0]
            item.match_type = "perfect"
            item.match_score = getattr(item, 'score', 0.95)
    
    elif match_type == "reverse":
        # Get a random item that is NOT in recommendations (Simulate reverse)
        # In a real system, we would ask InferenceService for lowest scores.
        # Here we pick a random item that is likely not high scoring.
        item = db.query(models.Item).filter(
            models.Item.id.notin_(swiped_subquery)
        ).order_by(func.random()).first()
        
        if item:
            item.match_type = "reverse"
            item.match_score = 0.1
            
    if not item:
        # Fallback if nothing found
        item = db.query(models.Item).filter(
            models.Item.id.notin_(swiped_subquery)
        ).order_by(models.Item.popularity.desc()).first()
        item.match_type = "none"
        
    # Map fields
    item.image_url = item.poster_path
    item.is_recommendation = True # Treat as recommendation for UI purposes
    
    return item