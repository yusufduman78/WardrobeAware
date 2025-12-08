from fastapi import FastAPI
from .database import engine, Base
from .routers import auth, feed, swipe

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Taste Match API")

app.include_router(auth.router)
app.include_router(feed.router)
app.include_router(swipe.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Taste Match API"}
