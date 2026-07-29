# References

## Projects cloned / adapted

1. **Driver-State-Detection** — https://github.com/e-candeloro/Driver-State-Detection  
   - Used for: EAR formula, attention timers, PERCLOS idea, sample `demo.mp4`  
   - License: MIT (see `third_party/Driver-State-Detection.LICENSE`)

2. **Gaze-Detection** — https://github.com/MohamedASAK/Gaze-Detection  
   - Used for: iris-in-eye horizontal/vertical ratio → gaze direction labels

## Datasets (for future training, optional)

- YawDD: https://www.site.uottawa.ca/~shervin/yawdd/
- MRL Eye: http://mrl.cs.vsb.cz/eyedataset
- UTA-RLDD: https://sites.google.com/view/utarldd/home
- NTHU Drowsy Driver Detection: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

## Problem mapping (FQC pilot)

Original request (excluding NG board placement):
- Check if operator is nodding off while inspecting boards → Sleep / Tired modules
- Check if operator is looking at the screen while checking → Eye direction / looking-at-screen
