from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas, database
from .auth import get_current_user

router = APIRouter(
    prefix="/swipe",
    tags=["swipe"]
)

@router.post("/")
def create_swipe(
    swipe: schemas.SwipeCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # Check if item exists
    item = db.query(models.Item).filter(models.Item.id == swipe.item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Check if already swiped
    existing_swipe = db.query(models.Swipe).filter(
        models.Swipe.user_id == current_user.id,
        models.Swipe.item_id == swipe.item_id
    ).first()
    
    if existing_swipe:
        raise HTTPException(status_code=400, detail="Already swiped on this item")

    new_swipe = models.Swipe(
        user_id=current_user.id,
        item_id=swipe.item_id,
        action=swipe.action
    )
    db.add(new_swipe)
    db.commit()
    
    # Not: Burada eskiden user vektörü güncellenirdi. 
    # Yeni sistemde vektörler inference anında hesaplandığı için (veya asenkron batch job ile)
    # burada anlık bir işlem yapmamıza gerek yok. Sadece kaydetmek yeterli.
    
    return {"message": "Swipe recorded"}