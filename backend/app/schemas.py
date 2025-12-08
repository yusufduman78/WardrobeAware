from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class ItemType(str, Enum):
    movie = "movie"
    game = "game"

class SwipeAction(str, Enum):
    like = "like"
    dislike = "dislike"
    superlike = "superlike"
    watchlist = "watchlist"

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class ItemOut(BaseModel):
    id: int
    type: str = "movie" # Enum yerine string daha esnek olabilir, hata vermemesi için
    title: str
    image_url: Optional[str] = None # Poster path buraya maplenecek
    overview: Optional[str] = None
    vote_average: Optional[float] = 0.0
    genres: Optional[str] = None
    is_recommendation: bool = False
    match_score: float = 0.0
    match_type: str = "none" # "perfect", "reverse", "none"
    
    # Frontend metadata_content bekliyorsa patlamasın diye ekleyelim ama içi boş olabilir
    metadata_content: Optional[Dict[str, Any]] = {}

    class Config:
        orm_mode = True

class SwipeCreate(BaseModel):
    item_id: int
    action: SwipeAction