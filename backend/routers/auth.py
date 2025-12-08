from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/auth", tags=["auth"])

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    avatar: Optional[str] = None

# Mock User Database
MOCK_USER = {
    "id": "user_123",
    "username": "fashionista",
    "email": "demo@example.com",
    "avatar": "https://i.pravatar.cc/150?u=fashionista",
    "superlikes": [],
    "watchlist": []
}

@router.get("/profile")
async def get_profile():
    # For MVP, we just return a mock user profile to simulate a logged-in state
    return MOCK_USER

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    # Mock Login - Accept any credentials
    return {
        "access_token": "mock_jwt_token_12345",
        "token_type": "bearer",
        "user": MOCK_USER
    }

class RegisterRequest(BaseModel):
    username: str
    email: Optional[str] = None
    password: str

@router.post("/register")
async def register(request: RegisterRequest):
    # Mock Register
    return {
        "access_token": "mock_jwt_token_new_user",
        "token_type": "bearer",
        "user": {
            "id": "user_new",
            "username": request.username,
            "email": request.email,
            "avatar": "https://i.pravatar.cc/150?u=new"
        }
    }
