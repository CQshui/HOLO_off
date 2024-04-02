from Fresnel_off import Fresnel_re
import os


direction = 'C:/Users/d1009/Desktop/test/origin/start'
file_names = os.listdir(direction)
files = [f for f in file_names if any(ext in f.lower() for ext in ('.jpg', '.jpeg', '.png', '.bmp'))]

for file in files:
    img_direction = os.path.join(direction, file)
    lamda = 532e-9
    pix = 0.098e-6
    z1 = 1e-5
    z2 = 3e-5
    z_interval = 1e-6
    Fresnel_re(img_direction, lamda, pix, z1, z2, z_interval, file)
