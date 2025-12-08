import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config
import torch
class PolyvoreTripletDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.transform = transform
        
        # Dosya yollarını config'den al
        json_file = config.TRAIN_JSON if split == 'train' else config.VALID_JSON
        
        print(f"Veri yükleniyor: {json_file}")
        with open(json_file, 'r') as f:
            self.outfits = json.load(f)
            
        with open(config.METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)
            
        # Kategori Haritalama (String -> Int ID)
        # Metadata'dan tüm semantic_category'leri bulup bir ID atıyoruz.
        self.cat_to_id = self._build_category_map()
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        print(f"Toplam Kategori Sayısı: {len(self.cat_to_id)}")

    def _build_category_map(self):
        categories = set()
        for item_id, data in self.metadata.items():
            if 'semantic_category' in data:
                categories.add(data['semantic_category'])
        # Alfabetik sıralayıp ID verelim ki her çalışmada aynı olsun
        return {cat: i for i, cat in enumerate(sorted(list(categories)))}

    def get_item_category(self, item_id):
        if item_id in self.metadata:
            return self.metadata[item_id].get('semantic_category', None)
        return None

    def __len__(self):
        return len(self.outfits)

    def load_img(self, item_id):
        # Metadata'dan dosya yolunu bul veya items klasöründen tahmin et
        # Genelde polyvore item_id ile dosya adı eşleşir ama metadata'da tam path olabilir.
        # Senin yapında: images/123456.jpg gibi duruyor olabilir.
        # Biz metadata'yı kontrol edelim, yoksa id.jpg varsayalım.
        
        # Not: Polyvore veri setinde bazen image pathler url olabilir, 
        # ama senin yerel klasöründe muhtemelen item_id.jpg şeklindedir.
        filename = f"{item_id}.jpg"
        path = os.path.join(config.IMAGES_ROOT, filename)
        
        # Eğer dosya yoksa, placeholder veya hata döndür (basitlik için hata fırlatma, random siyah resim dön)
        if not os.path.exists(path):
            # Try finding path from outfit item data if available, otherwise skip
            return Image.new('RGB', (224, 224))
            
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        # 1. Anchor Seçimi
        outfit = self.outfits[idx]
        items = outfit['items']
        
        # Geçerli (resmi ve kategorisi olan) itemları filtrele
        valid_items = [i for i in items if self.get_item_category(i['item_id']) in self.cat_to_id]
        
        if len(valid_items) < 2:
            # Kombinde yeterli parça yoksa başka bir outfit seç
            return self.__getitem__(random.randint(0, len(self.outfits)-1))
            
        anchor = random.choice(valid_items)
        anchor_id = anchor['item_id']
        anchor_cat_str = self.get_item_category(anchor_id)
        anchor_cat_idx = self.cat_to_id[anchor_cat_str]
        
        # 2. Positive Seçimi (Aynı kombinden, FARKLI kategori)
        possibles = [i for i in valid_items if self.get_item_category(i['item_id']) != anchor_cat_str]
        
        if not possibles:
            return self.__getitem__(random.randint(0, len(self.outfits)-1))
            
        pos = random.choice(possibles)
        pos_id = pos['item_id']
        pos_cat_str = self.get_item_category(pos_id)
        pos_cat_idx = self.cat_to_id[pos_cat_str]
        
        # 3. Negative Seçimi (Rastgele başka bir kombinden, POSITIVE ile AYNI kategori - Hard Negative)
        # Döngüye girmemek için max deneme sayısı koyalım
        neg_item = None
        for _ in range(10):
            rand_idx = random.randint(0, len(self.outfits)-1)
            rand_outfit = self.outfits[rand_idx]
            rand_items = rand_outfit['items']
            
            # Positive ile aynı kategoride olan bir item arıyoruz
            candidates = [i for i in rand_items if self.get_item_category(i['item_id']) == pos_cat_str]
            if candidates:
                neg_item = random.choice(candidates)
                break
        
        if neg_item is None:
            # Eğer bulamazsak, rastgele herhangi bir item alalım (Fallback)
            rand_idx = random.randint(0, len(self.outfits)-1)
            rand_outfit = self.outfits[rand_idx]
            if rand_outfit['items']:
                 neg_item = random.choice(rand_outfit['items'])
                 # Kategori uyuşmazsa da devam et, eğitim durmasın
        
        neg_id = neg_item['item_id']
        # Eğer neg'in kategorisi yoksa (metadata eksikse), pos kategorisini varsayalım (riskli ama kod çalışır)
        neg_cat_str = self.get_item_category(neg_id)
        neg_cat_idx = self.cat_to_id.get(neg_cat_str, pos_cat_idx)

        # Görüntüleri Yükle
        img_a = self.load_img(anchor_id)
        img_p = self.load_img(pos_id)
        img_n = self.load_img(neg_id)
        
        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)
            
        return (img_a, img_p, img_n, 
                torch.tensor(anchor_cat_idx), torch.tensor(pos_cat_idx), torch.tensor(neg_cat_idx))