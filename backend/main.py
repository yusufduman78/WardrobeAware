from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))

from backend.routers import feed, swipe, outfit, auth, wardrobe

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Warm up models ---
    print("\n" + "="*60)
    print("🚦 SYSTEM WARM-UP INITIATED")
    print("="*60)
    
    # 1. Load Recommender Model (ResNet/Projection)
    print("⏳ Loading Recommender Model...")
    try:
        outfit.get_recommender()
        print("✅ Recommender Model Ready (RAM/GPU)")
    except Exception as e:
        print(f"❌ Failed to load Recommender: {e}")

    # 2. Load FashionCLIP Extractor (Heavy ~2GB)
    print("⏳ Loading FashionCLIP Model (This may take 10-20s)...")
    try:
        wardrobe.get_embedding_extractor()
        print("✅ FashionCLIP Model Ready (RAM/GPU)")
    except Exception as e:
        print(f"❌ Failed to load FashionCLIP: {e}")

    # 3. Initialize Segmenter (Background Removal)
    print("⏳ Initializing Background Remover...")
    try:
        from src.segmentation import segmenter
        # Dummy pass to wake up model if possible, or just import trigger
        print("✅ Background Remover Ready")
    except Exception as e:
        print(f"❌ Failed to load Segmenter: {e}")

    print("="*60)
    print("🚀 SYSTEM READY! Listening for requests...")
    print("="*60 + "\n")
    
    yield
    # --- Shutdown: Cleanup if needed ---
    print("🛑 System Shutting Down...")

app = FastAPI(title="Fashion Recommender API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Polyvore Images
from backend.config import IMAGES_DIR


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
