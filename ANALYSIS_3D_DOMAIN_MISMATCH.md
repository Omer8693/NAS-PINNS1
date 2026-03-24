# NAS-PINNS3: 3D Domain ve Plot Uyumsuzluğu Analizi

## 🎯 Sorun Özeti

Web arayüzü (demo.html) dört domain gösteriyor: rectangular, cylinder, stacked, lshape. 
Ancak 3D yüzey visualizasyonunda tüm domainler **düz z-kesiti** olarak çiziliyorken, 
özellikle **L-Shape domain'in gerçek geometrisi gösterilmiyor**:

- **Rectangular (✓)**: 1.3×0.6×0.4 m dikdörtgen prizm - DOĞRU gösteriliyor
- **Cylinder (⚠️)**: Silindir olması gereken ama düz dikdörtgen xy-kesiti gösteriliyor
- **Stacked (⚠️)**: 2 küp yığını olması gereken ama düz kesit gösteriliyor  
- **L-Shape (❌)**: **ALINAN KÖŞESİ gösterilmiyor** - standart rectangular gibi gösteriliyor

---

## 📋 Teknik Analiz

### Problem 1: 3D Visualization Stratejisi Yanlış

**Dosya**: `/home/coder/NAS-PINNS1/NAS-PINNS3/web/static/js/demo.js` (Satır ~270-290)

```javascript
function render3D() {
  const { z: Tmat, zmin, zmax, label } = getCurrentZ(state.window);
  const d = state.sliceData;
  
  // PROBLEM: Tüm domainler için statik düz z-kesiti oluşturuluyor
  const flatZ = Array.from({length: ny}, () => Array(nx).fill(d.z_val));
  
  Plotly.react("plot-3d", [{
    type: "surface",
    x: d.xi, y: d.yi, z: flatZ,        // ← z-değer DÜZ, gerçek 3D geometri yok
    surfacecolor: Tmat,
  }]);
}
```

**Sonuç**: 
- Silindir = dikdörtgen (çember) 
- L-Shape = dikdörtgen (köşe kesik değil)
- Stacked = tek kesit

---

### Problem 2: L-Shape Domain Tanımlaması

**Dosya**: `/home/coder/NAS-PINNS1/NAS-PINNS3/level8_nas_mco_pinn/domains_3d.py` (Satır ~306)

L-Shape geometrisi:
```python
class LShape3D:
    """
    Cross-section (xy) = two rectangular arms joined at corner:
        mask: (x <= cut_x) OR (y <= cut_y)   [for all z in [0, Lz]]
    
    Domain dimensions:
        Lx=0.8m, Ly=0.8m, Lz=0.4m
        cut_x=0.3m, cut_y=0.3m  (remove top-right corner)
    """
    
    def _mask_3d(self, x, y, z):
        """L-shape: Sağ-üst çeyrek kullanılamaz"""
        return (((x <= self.cut_x) | (y <= self.cut_y)) &
                (z >= 0) & (z <= self.Lz))
```

**Gerçek alan**: L-şekli (0.8×0.8 alalanda 0.3×0.3 çeyrek kesili)
**Şu anda görünen**: Tam dikdörtgen 0.8×0.8

---

### Problem 3: Veri Yapısındaki Eksiklik

**Dosya**: `/home/coder/NAS-PINNS1/NAS-PINNS3/run_3d_v2.py` (Satır ~75-95)

Kaydedilen veri yapısında SADECE dikdörtgen alan için:
```json
{
  "xi": [x1, x2, ...],    // 1D array
  "yi": [y1, y2, ...],    // 1D array
  "zi": [z1, z2, ...],    
  "k_mid": 10,
  "z_val": 0.2,
  "windows": {...}        // T_pred ve T_fem 2D arrays
}
```

**EKSIK**: 
- Domain masking bilgisi (hangi xy noktaları geçerli?)
- Geometry metadata (cylinder radius, L-shape cuts, etc.)

---

## ✅ Çözüm Planı

### Adım 1️⃣: JSON Veri Yapısını Genişlet
Dosya: `run_3d_v2.py` ve data generation

Her domain için metadata ekle:
```json
{
  "xi": [...],
  "yi": [...],
  "zi": [...],
  "z_val": 0.2,
  "geometry": {
    "type": "lshape",           // ← domain tipi
    "params": {
      "Lx": 0.8,
      "Ly": 0.8,
      "Lz": 0.4,
      "cut_x": 0.3,
      "cut_y": 0.3
    }
  },
  "windows": {...}
}
```

### Adım 2️⃣: 3D Rendering'i Domain-Spesifik Yap
Dosya: `demo.js`

Domain türüne göre farklı rendering çıktıları:

```javascript
function render3D() {
  const { z: Tmat, ... } = getCurrentZ(state.window);
  const d = state.sliceData;
  const geom = d.geometry || {};
  
  let zPlane, xyMask;
  
  if (geom.type === "rectangular") {
    // Flat plane
    zPlane = Array(...).fill(d.z_val);
    xyMask = null;
  } 
  else if (geom.type === "cylinder") {
    // Cylindrical surface at z_val
    const R = geom.params.R;
    zPlane = buildCylinderSurface(d.xi, d.yi, d.z_val, R);
    xyMask = buildCylinderMask(d.xi, d.yi, R);
  }
  else if (geom.type === "lshape") {
    // L-shaped flat plane with masked region
    zPlane = Array(...).fill(d.z_val);
    xyMask = buildLShapeMask(d.xi, d.yi, 
      geom.params.cut_x, geom.params.cut_y);
  }
  // ... etc
  
  // Plotly'de surfacecolor maskelemeyle göster
  const maskedColor = applyMask(Tmat, xyMask);
  
  Plotly.react("plot-3d", [{...}]);
}
```

### Adım 3️⃣: Frontend Validation
Dosya: `demo.js`

Domain yüklenirken hata kontrolü:
```javascript
function loadData() {
  // ... fetch...
  Promise.all([...]).then(([slice, loss]) => {
    if (!slice.geometry) {
      console.warn("Geometry metadata eksik - eski veri formatı?");
    }
    state.sliceData = slice;
    // ...
  });
}
```

### Adım 4️⃣: Backend Güncelle
Dosya: `run_3d_v2.py`

```python
# Her domain için metadata ekleme
def save_slice_data(domain, grid, slices, skip, out_dir):
    geom_types = {
        'Rectangular3D': 'rectangular',
        'Cylinder3D': 'cylinder',
        'StackedCubes3D': 'stacked',
        'LShape3D': 'lshape'
    }
    
    slice_data = {
        "xi": grid["xi"].tolist(),
        "yi": grid["yi"].tolist(),
        "zi": grid["zi"].tolist(),
        "z_val": float(grid["zi"][grid["k_mid"]]),
        "geometry": {
            "type": geom_types[type(domain).__name__],
            "params": extract_geometry_params(domain)
        },
        "windows": slices
    }
```

---

## 🔧 Kaynaklar Dosyalar

| Dosya | Sorun | Çözüm |
|-------|-------|-------|
| `web/static/js/demo.js` | `render3D()` sadece flat plane çiziyor | Domain-spesifik geometri çizme logic |
| `run_3d_v2.py` | Geometry metadata kaydedilmiyor | JSON'a `geometry` field ekle |
| `web/templates/demo.html` | Açıklamalar generic | Domain açıklamalarını güncelle |
| `level8_nas_mco_pinn/domains_3d.py` | LShape3D tanımlı ama metadata yok | Export helper function |

---

## 📊 Doğrulama Checklist

- [ ] L-Shape domain'in köşesi (removed quadrant) gösterililiyor
- [ ] Silindir gerçek silindir yüzeyi olarak gösteriliyor  
- [ ] Geometry metadata her domain için JSON'da var
- [ ] Browser console'da uyarı yok
- [ ] Tüm 4 domain (rectangular, cylinder, stacked, lshape) çalışıyor
- [ ] 2D heatmap'ler de masked region'ları gösteriyor (L-Shape için)

---

## 🚀 İmlementasyon Sırası

1. **ÖNCE**: `run_3d_v2.py` - geometry metadata ekle ve yeniden çalıştır
2. **SONRA**: `demo.js` - render3D logic'i güncelle
3. **VERIFY**: Web'de L-Shape'in kesik köşesi görüntülensin
4. **NICE-TO-HAVE**: Cylinder ve Stacked için de gerçek geometri çiz

