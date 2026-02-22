import os
import time

while True:
    os.system('git add .')
    os.system('git commit -m "auto commit"')
    os.system('git push')
    time.sleep(1800)  # 30 dakika = 1800 saniye