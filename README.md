# WardrobeAware

An AI-powered fashion recommendation system that learns your style. Features outfit generation, compatibility scoring, and wardrobe digitization.

## 🚀 Quick Start Guide

Follow these steps exactly to run the project.

### 1. Prerequisites
- **Python 3.10+**: [Download](https://www.python.org/downloads/)
- **Node.js 18+**: [Download](https://nodejs.org/)
- **Git LFS**: [Download](https://git-lfs.com/) (Required for models)

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yusufduman78/WardrobeAware.git
cd WardrobeAware

# Pull LFS files (Important for models!)
git lfs pull

# Create Virtual Environment (mamba/conda recommended)
conda create -n wardrobeaware python=3.10
conda activate wardrobeaware

# Install Python Dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install fashion-clip
```

### 3. Data Setup (Crucial!)
You must download the **Polyvore Outfits Dataset** separately as it is too large for GitHub.
1. Download the dataset.
2. Extract it to: `data/polyvore_outfits/`
3. Ensure this exact structure:
   ```
   WardrobeAware/
   ├── data/
   │   └── polyvore_outfits/
   │       ├── images/               # Contains thousands of .jpg files
   │       └── polyvore_item_metadata.json
   ```

### 4. Configuration (Centralized IP)
Updates are needed in **only two files** to match your network IP.

**Step 1:** Open `backend/config.py`
```python
HOST_IP = "192.168.1.6"  # <--- Change this to your computer's LAN IP
```

**Step 2:** Open `tastematch/constants/Config.ts`
```typescript
const API_URL = 'http://192.168.1.6:8000'; // <--- Update IP here too
```

### 5. Install Frontend Dependencies
```bash
cd tastematch
# If you are missing the assets folder, ensure you copy it from your source!
npm install
cd ..
```

### 6. Run the Application

**Terminal 1: Backend**
```bash
# Make sure you are in the root 'WardrobeAware' folder
conda activate wardrobeaware
python run_server.py
# Verify it says "Images directory mounted..."
```

**Terminal 2: Frontend**
```bash
cd tastematch
npx expo start
# Scan the QR code with Expo Go app or press 'a' for Android Emulator
```

## 📂 Project Structure
- `backend/`: FastAPI server
- `tastematch/`: React Native mobile app
- `recommender/`: ML models and logic
- `models/`: Trained checkpoints (managed by Git LFS)

## 🐛 Troubleshooting
- **Images not loading?** Double check `backend/config.py` has the right `HOST_IP` and `data/polyvore_outfits/images` exists.
- **App crash on launch?** Ensure `tastematch/assets` folder is present.
- **Model error?** Run `git lfs pull` again to ensure `model_epoch_15.pth` is fully downloaded (approx 100MB+).
