import pandas as pd
import numpy as np
from app.database import SessionLocal, engine, Base
from app.models import Item
import os

# DOSYA YOLLARI
LINKS_PATH = "../project/data/ml-latest/links.csv"
TMDB_PATH = "data/tmdb-movies.csv" 

def ingest_simple():
    # Temiz kurulum (Tabloları sıfırla)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    print("Dosyalar okunuyor...")
    
    # 1. Links Dosyasını "Sözlük" Olarak Hazırla
    # Amacımız: tmdb_id verince ml_id almak.
    links_df = pd.read_csv(LINKS_PATH, dtype={'movieId': int, 'tmdbId': str})
    links_df = links_df.dropna(subset=['tmdbId'])
    # "862.0" gibi gelenleri "862" yap
    links_df['tmdbId'] = links_df['tmdbId'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # { '862': 1, '8844': 2 ... } formatında sözlük
    tmdb_to_ml = dict(zip(links_df['tmdbId'], links_df['movieId']))
    print(f"Link sözlüğü hazır: {len(tmdb_to_ml)} eşleşme var.")

    # 2. TMDB Verisini Oku
    movies_df = pd.read_csv(TMDB_PATH, dtype={'id': str})
    print(f"TMDB verisi okundu: {len(movies_df)} satır.")
    
    # 3. YENİ SÜTUN EKLEME (Senin dediğin yöntem)
    # 'id' sütunundaki değere bak, sözlükte karşılığı varsa 'ml_id'ye yaz.
    print("ML ID'leri eşleştiriliyor...")
    movies_df['ml_id'] = movies_df['id'].map(tmdb_to_ml)
    
    # 4. TEMİZLİK
    # ml_id'si olmayan (yani modelin tanımadığı) filmleri at.
    movies_df = movies_df.dropna(subset=['ml_id'])
    # ml_id float olmuştur (NaN yüzünden), int'e çevir
    movies_df['ml_id'] = movies_df['ml_id'].astype(int)
    
    # DUPLICATE KONTROLÜ (Az önceki hatayı önler)
    # Aynı ml_id'ye sahip birden fazla satır varsa ilkini tut, diğerlerini at.
    movies_df = movies_df.drop_duplicates(subset=['ml_id'])
    
    # Popülerliğe göre sırala, ilk 25.000'i al
    if 'popularity' in movies_df.columns:
        movies_df = movies_df.sort_values(by='popularity', ascending=False)
    
    movies_df = movies_df.head(25000)
    print(f"Veritabanına yazılacak net film sayısı: {len(movies_df)}")

    # 5. VERİTABANINA YAZMA
    base_image_url = "https://image.tmdb.org/t/p/w780"
    items_to_add = []
    count = 0
    
    for _, row in movies_df.iterrows():
        try:
            # Resim yolları
            poster = f"{base_image_url}{row['poster_path']}" if pd.notna(row.get('poster_path')) else None
            backdrop = f"{base_image_url}{row['backdrop_path']}" if pd.notna(row.get('backdrop_path')) else None
            
            item = Item(
                ml_id=int(row['ml_id']),      # Eşleştirdiğimiz ID
                tmdb_id=str(row['id']),
                external_id=f"tmdb_{row['id']}",
                type="movie",
                title=str(row.get('title', row.get('original_title', 'Unknown'))),
                overview=str(row.get('overview', '')),
                genres=str(row.get('genres', '')),
                poster_path=poster,
                backdrop_path=backdrop,
                vote_average=float(row.get('vote_average', 0)),
                vote_count=int(row.get('vote_count', 0)),
                popularity=float(row.get('popularity', 0)),
                release_date=str(row.get('release_date', ''))
            )
            items_to_add.append(item)
            count += 1
            
            if len(items_to_add) >= 1000:
                db.bulk_save_objects(items_to_add)
                db.commit()
                items_to_add = []
                print(f"{count} film eklendi...")
                
        except Exception as e:
            print(f"Hata: {e}")
            continue
            
    if items_to_add:
        db.bulk_save_objects(items_to_add)
        db.commit()
        
    db.close()
    print("Bitti! Veritabanı hem resimli hem de ML ID'li filmlerle dolu.")

if __name__ == "__main__":
    ingest_simple()