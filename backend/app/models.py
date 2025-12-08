from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime
import enum

class SwipeAction(str, enum.Enum):
    like = "like"
    dislike = "dislike"
    superlike = "superlike"
    watchlist = "watchlist"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # Modelden gelen embedding vektörünü önbelleklemek istersek diye (şimdilik boş durabilir)
    embedding = Column(JSON, default=[]) 
    
    swipes = relationship("Swipe", back_populates="user")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    
    # --- YENİ MODEL İÇİN GEREKLİ ALANLAR ---
    ml_id = Column(Integer, index=True, nullable=True) # Modelin bildiği ID
    tmdb_id = Column(String, index=True, nullable=True) # TMDB ID
    
    type = Column(String, default="movie")
    external_id = Column(String, unique=True, index=True) # tmdb_123
    title = Column(String)
    overview = Column(String)
    genres = Column(String)
    
    # Görseller
    poster_path = Column(String)   # Dikey Afiş (https://image.tmdb...)
    backdrop_path = Column(String) # Yatay Kapak
    
    vote_average = Column(Float, default=0.0)
    vote_count = Column(Integer, default=0)
    popularity = Column(Float, default=0.0)
    release_date = Column(String, nullable=True)
    
    metadata_content = Column(JSON, nullable=True) # Ekstra veriler için

    swipes = relationship("Swipe", back_populates="item")

class Swipe(Base):
    __tablename__ = "swipes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_id = Column(Integer, ForeignKey("items.id"))
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="swipes")
    item = relationship("Item", back_populates="swipes")