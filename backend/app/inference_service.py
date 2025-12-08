import os
import pickle
import numpy as np
from .database import SessionLocal
from .models import Item, Swipe, SwipeAction

# Model dosya yollarını ayarla (Project klasörüne gidip alacak)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "project", "models", "fast_svd_model.pkl")
MAPPING_PATH = os.path.join(BASE_DIR, "project", "models", "mappings.pkl")

# Pickle 'models' modülünü arayacağı için path'e ekle
import sys
sys.path.append(os.path.join(BASE_DIR, "project", "src"))

class InferenceService:
    def __init__(self):
        self.model = None
        self.mappings = None
        self.item_factors = None
        self.initialized = False
        
    def load_model(self):
        if self.initialized: return
        print("Model yükleniyor...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
                # Ensure item_factors is (N_items, 64)
                # If shape is (64, N), transpose it. If (N, 64), keep it.
                if self.model.item_vecs.shape[0] < self.model.item_vecs.shape[1]:
                     self.item_factors = self.model.item_vecs.T
                else:
                     self.item_factors = self.model.item_vecs
            
            with open(MAPPING_PATH, 'rb') as f:
                self.mappings = pickle.load(f)
                
            self.initialized = True
            print(f"Model Hazır. Item Factors Shape: {self.item_factors.shape}")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")

    def get_recommendations(self, user_id: int, limit=20):
        if not self.initialized: self.load_model()
        
        # Eğer model yüklenemediyse boş liste dön (500 hatası verme)
        if not self.initialized or self.mappings is None:
            print("Model yüklenemediği için öneri yapılamıyor.")
            return []

        db = SessionLocal()
        
        # 1. Kullanıcının beğendiği (like/superlike) filmlerin ml_id'lerini çek
        liked_rows = db.query(Item.ml_id).join(Swipe).filter(
            Swipe.user_id == user_id,
            Swipe.action.in_([SwipeAction.like, SwipeAction.superlike]),
            Item.ml_id.isnot(None)
        ).all()
        
        liked_ml_ids = [r[0] for r in liked_rows]
        
        # 2. Kullanıcı Vektörü Oluştur
        user_vector = np.zeros(64) # 64 boyutlu eğitmiştik
        count = 0
        
        for mid in liked_ml_ids:
            # Modelin tanıdığı ID'ye (index) çevir
            if mid in self.mappings['movie2idx']:
                idx = self.mappings['movie2idx'][mid]
                # Item vektörünü ekle
                user_vector += self.item_factors[idx]
                count += 1
                
        if count > 0:
            user_vector /= count # Ortalama al
        else:
            # Soğuk başlangıç: Hiçbir şey beğenmediyse popülerleri döndür
            # (Bunu basitçe boş liste dönerek halledelim, router popülerleri doldursun)
            db.close()
            return []

        # 3. Skor Hesapla (Dot Product)
        # User (64,) . Items (N, 64).T -> (N,)
        # OR np.dot(Items, User) -> (N, 64) . (64,) -> (N,)
        scores = np.dot(self.item_factors, user_vector)
        
        # 4. En yüksek skorlu indexleri bul
        top_indices = scores.argsort()[::-1][:limit*3] # Filtrelemek için fazla al
        
        recommended_ml_ids = []
        for idx in top_indices:
            real_ml_id = self.mappings['idx2movie'][idx]
            
            # Zaten izlediklerini önerme
            if real_ml_id not in liked_ml_ids:
                recommended_ml_ids.append(real_ml_id)
                if len(recommended_ml_ids) >= limit:
                    break
        
        # 5. DB'den Itemları Çek (ml_id'ye göre)
        items = db.query(Item).filter(Item.ml_id.in_(recommended_ml_ids)).all()
        
        # Sıralamayı koru (SQL 'IN' sorgusu sıralamayı bozar)
        item_map = {item.ml_id: item for item in items}
        sorted_items = []
        
        # Score map oluştur (ml_id -> score)
        score_map = {}
        for idx in top_indices:
             real_ml_id = self.mappings['idx2movie'][idx]
             score_map[real_ml_id] = float(scores[idx]) # float'a çevir

        for mid in recommended_ml_ids:
            if mid in item_map:
                item = item_map[mid]
                # Skoru item üzerine ekle (geçici olarak)
                item.score = score_map.get(mid, 0.0)
                sorted_items.append(item)
                
        db.close()
        return sorted_items

inference_engine = InferenceService()