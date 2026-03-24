# NAS-PINNS3: 3D Domain Mismatch - Çözüm Uygulaması

## 🎯 Yapılan Değişiklikler

### 1️⃣ Backend: `run_3d_v2.py`

#### Değişiklik A: LShape3D import'u ekleme
```python
from level8_nas_mco_pinn.domains_3d import Rectangular3D, Cylinder3D, StackedCubes3D, LShape3D
```

#### Değişiklik B: Geometry metadata ekstraksiyon helper
Yeni bir helper fonksiyon eklendi:
```python
def get_geometry_metadata(domain):
    """Extract geometry parameters from domain object."""
    # Returns dict with type, params for each domain
```

Bu fonksiyon:
- `Rectangular3D`: `{type: "rectangular", params: {Lx, Ly, Lz}}`
- `Cylinder3D`: `{type: "cylinder", params: {R, H}}`  
- `StackedCubes3D`: `{type: "stacked", params: {L_cube, N_stack, Lz}}`
- `LShape3D`: `{type: "lshape", params: {Lx, Ly, Lz, cut_x, cut_y}}`

#### Değişiklik C: DOMAINS dict güncelleme
```python
DOMAINS = {
    "rectangular": Rectangular3D(),
    "cylinder":    Cylinder3D(),
    "stacked":     StackedCubes3D(),
    "lshape":      LShape3D(),  # ← YENİ
}
```

#### Değişiklik D: JSON veri yapısı genişletme
Kaydedilen slice veri yapısına `geometry` field eklendi:
```json
{
  "xi": [...],
  "yi": [...],
  "zi": [...],
  "z_val": 0.2,
  "geometry": {
    "type": "lshape",
    "params": {Lx, Ly, Lz, cut_x, cut_y}
  },
  "windows": {...}
}
```

---

### 2️⃣ Frontend: `web/static/js/demo.js`

#### Değişiklik A: L-Shape maskelemesi fonksiyonları
İki yeni helper fonksiyon eklendi:

```javascript
function buildLShapeMask(xi, yi, cut_x, cut_y)
  /* L-shape dışındaki bölgeleri NaN ile işaretler */

function applyMaskToValues(values, mask)
  /* Değerleri maske ile çarparak görünümü kontrol eder */
```

#### Değişiklik B: Domain-spesifik 3D rendering
`render3D()` fonksiyonu artık:
- Geometry metadata'yı okur
- Domain tipine göre farklı maskeleme uygular:
  - **L-Shape**: `buildLShapeMask()` ile köşe kesintisi gösterir
  - **Cylinder**: Dairesel maske uygular
  - **Rectangular/Stacked**: Düz kesit (değişmedi)

#### Değişiklik C: Domain-spesifik 2D rendering
`render2D()` fonksiyonu artık:
- L-Shape için şekil (shape) ekler kesik bölgeyi göstermek için
- Ek metin annotation'ı ekler (cut parametreleri vs.)
- Cylinder için annotation'da radius gösterir

#### Değişiklik D: Veri formatı uyarısı
`loadData()` fonksiyonuna uyarı eklendi:
```javascript
if (!slice.geometry) {
  console.warn("⚠️  Geometry metadata not found...");
}
```

Eski veri formatı (geometry metadata yok) kullanılırsa browser console'da uyarı gösterilir.

---

## 📋 Doğrulama Adımları

### Step 1: Python Syntax Kontrolü
```bash
cd /home/coder/NAS-PINNS1/NAS-PINNS3
python3 -m py_compile run_3d_v2.py
# Hata yok ise OK
```

### Step 2: Data Yeniden Oluşturma (İsteğe bağlı)
Eğer training çalıştırılacaksa:
```bash
cd /home/coder/NAS-PINNS1/NAS-PINNS3
python3 run_3d_v2.py
# Tetmelde: new geometry field, lshape data
```

Örnek output:
```
  Domain: Rectangular Prism 3D
  Domain: Cylinder 3D
  Domain: Stacked Cubes 3D
  Domain: L-Shape 3D          ← YENİ
```

### Step 3: JSON Veri Doğrulama
Oluşturulan veri dosyalarını kontrol et:
```bash
ls /home/coder/NAS-PINNS1/NAS-PINNS3/level8_nas_mco_pinn/results/v2/ | grep lshape
# lshape_bayesian_skip1_slice.json
# lshape_bayesian_skip2_slice.json
# ... vb.

# Metadata var mı kontrol et:
python3 << 'EOF'
import json
with open("level8_nas_mco_pinn/results/v2/lshape_bayesian_skip2_slice.json") as f:
    data = json.load(f)
    print("Has geometry:", "geometry" in data)
    if "geometry" in data:
        print("Type:", data["geometry"]["type"])
        print("Params:", data["geometry"]["params"])
EOF
```

### Step 4: Web Interface Test
```bash
cd /home/coder/NAS-PINNS1/NAS-PINNS3/web
python3 app.py &
# Browser: http://localhost:5000/demo
```

**Kontrol listesi:**
- [ ] Lshape domain seçildiğinde yükleniyor
- [ ] 2D heatmap'te kesik bölge gösteriliyoru
- [ ] 3D surface'te kesik bölge gösteriliyoru (NaN regions)
- [ ] Cylinder seçildiğinde merkez etrafındaki daire gösteriliyoru
- [ ] Annotation'lar geometri parametrelerine sahiptir
- [ ] Browser console'da hata yok

### Step 5: Eski Veri Uyumluluğu (Graceful Degradation)
Eski veri (geometry field yok) kullanılırsa:
- [ ] Hala yükleniyor
- [ ] Console'da **⚠️ warning** gösteriliyor
- [ ] Default rectangular gösteriliyor
- [ ] Crash yok

---

## 🚀 Dağıtım Kılavuzu

### Seçenek A: Mevcut Veriye Karşı Çalıştırma (Uyumluluk Modu)
Varolan veri dosyalarına sahip devam etmek isterseniz:
1. Demo web arayüzünü açın
2. Console'da uyarıyı göreceksiniz
3. L-Shape hala yüklenecek (ama varsayılan rectangular olarak gösterilecek)
4. Geometry metadata olmadan daha az sonuç gösterilecek

### Seçenek B: Temiz Dağıtım (Önerilen)
Yeni veri oluşturmak isterseniz:
```bash
# Eski v2 verilerini yedekle
mv /home/coder/NAS-PINNS1/NAS-PINNS3/level8_nas_mco_pinn/results/v2 \
   /home/coder/NAS-PINNS1/NAS-PINNS3/level8_nas_mco_pinn/results/v2_old

# JSON'lar silinsin ama checkpoint'ler saklanabilir
mkdir /home/coder/NAS-PINNS1/NAS-PINNS3/level8_nas_mco_pinn/results/v2

# Yeniden çalıştır
python3 run_3d_v2.py

# Veriye bağlı olarak 30 dakika - 2 saat sürebilir
```

---

## ✅ Hata İzleme

| Sorun | Çözüm |
|-------|-------|
| `AttributeError: 'LShape3D' has no attribute 'Lx'` | LShape3D'de parametreler self'e atandı mı? |
| Browser console: "geometry is undefined" | Yeni data ile çalışırken sync edin |
| L-Shape kesik bölgesi gösterilmiyor | `buildLShapeMask()` function'ı doğru parametrelerle mi çalışıyor? |
| Web yükleme hatası 404 | lshape_{arch}_skip{s}_slice.json var mı? |
| Cylinder dairesel değil | Cylinder mask'ı merkez koordinatlarını doğru hesaplıyor mu? |

---

## 📊 Beklenen Sonuçlar

### Öncesi (Hatalı):
- ✗ L-Shape düz dikdörtgen olarak gösteriliyordu
- ✗ Cylinder dikdörtgen kesit olarak gösteriliyordu
- ✗ Geometry metadata yoktu

### Sonrası (Düzeltildi):
- ✅ L-Shape'in kesik köşesi 2D ve 3D'de görüntüleniyoru
- ✅ Cylinder'in dairesel maskeleme uygulanıyoru
- ✅ Her domain tipi doğru annotation'ları gösteriyor
- ✅ Geometry metadata JSON'da tutuluyoru
- ✅ Eski veri hala uyumlu (warning ile)

---

## 📝 Kaynak Dosyalar

| Dosya | Satırlar | Değişiklik |
|-------|---------|-----------|
| `run_3d_v2.py` | 1-50 | Import + LShape3D ekleme |
| `run_3d_v2.py` | 28-70 | `get_geometry_metadata()` fonksiyon |
| `run_3d_v2.py` | 72-78 | DOMAINS dict'e lshape ekle |
| `run_3d_v2.py` | 135-145 | JSON'a geometry field ekle |
| `demo.js` | 252-300 | Maskeleme helper fonksiyonları |
| `demo.js` | 301-370 | Domain-aware render3D() |
| `demo.js` | 211-290 | Şekil/annotation'lu render2D() |
| `demo.js` | 152-158 | Geometry metadata uyarımı |

---

## 🎓 Teknik Notlar

### Neden NaN Kullanıyoruz?
Plotly, NaN değerlerini otomatik olarak göstermez. Bu, dışarıdaki bölgeleri maskelemek için ideal:
```javascript
// Dışarı = NaN, İçeri = gerçek değer
// Plotly sadece gerçek değerleri çizer
colorData[j][i] = isInside ? Tmat[j][i] : NaN;
```

### L-Shape Maske Mantığı
```
Mask = (x <= cut_x) OR (y <= cut_y)

Örnek (cut_x=0.3, cut_y=0.3):
  [0,0]────────────[0.8,0]
    │ ✓✓✓✓✓✓│ ✗✗✗ │
    │ ✓✓✓✓✓✓│ ✗✗✗ │  ✓ = İçeri (show)
    │ ─────────────── │  ✗ = Dışarı (hide)
    │ ✓✓✓✓✓✓	 ✗✗✗ │
  [0,0.8]─────[0.3,0.8]
```

### Cylinder Maskelemesi
```javascript
const r = sqrt(x^2 + y^2);  // merkeze uzaklık
const inside = r <= R;       // daire içinde mi?
```

---

## 🔔 Sonraki Adımlar (Opsiyonel)

1. **3D Mesh Geometry**: Silindir/L-Shape için gerçek 3D yüzey mesh'leri oluştur
2. **Zeit Animation**: Time window'lar arasında Z-düzlemi animasyon yap
3. **Domain Karşılaştırması**: Domainler arasında MAE'yi grafik olarak karşılaştır
4. **Boundary Highlighting**: Robin BC'leri göster
