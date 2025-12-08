from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))

from backend.routers import feed, swipe, outfit, auth, wardrobe

app = FastAPI(title="Fashion Recommender API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Polyvore Images
# Assuming data/polyvore_outfits/images exists
IMAGES_DIR = Path(__file__).parent.parent / "data" / "polyvore_outfits" / "images"
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
    print(f"INFO: Images directory mounted at /images from {IMAGES_DIR}")
    # Check if directory has files
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    print(f"INFO: Found {len(image_files)} image files in directory")
    
    # Also mount user_uploads subdirectory
    USER_UPLOADS_DIR = IMAGES_DIR / "user_uploads"
    if USER_UPLOADS_DIR.exists():
        print(f"INFO: User uploads directory exists at {USER_UPLOADS_DIR}")
    else:
        USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Created user uploads directory at {USER_UPLOADS_DIR}")
else:
    print(f"WARNING: Image directory not found at {IMAGES_DIR}")

# Include Routers
app.include_router(feed.router)
app.include_router(swipe.router)
app.include_router(outfit.router)
app.include_router(auth.router)
app.include_router(wardrobe.router)

@app.get("/")
def read_root():
    return {"message": "Fashion Recommender API is running"}
