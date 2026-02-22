import os
import time
from datetime import datetime

# Kaç dakikada bir commit yapılacağını ayarlayın
ARALIK_DAKIKA = 30  # İstediğiniz aralığı buradan değiştirebilirsiniz

while True:
    # results/ klasöründeki tüm değişiklikleri ekle
    os.system("git add results/")
    # Commit mesajı olarak zaman bilgisini ekle
    commit_msg = f'Sonuçlar otomatik commit {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    os.system(f'git commit -m "{commit_msg}"')
    # Değişiklikleri mevcut branch'e gönder
    os.system("git push")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Commit ve push işlemi yapıldı.")
    # Belirtilen süre kadar bekle
    time.sleep(ARALIK_DAKIKA * 60)
