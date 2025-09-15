# PPG Stress Preprocessing

A project focusing on pre-processing PPG signals for stress detection. Includes filtering, noise removal, and signal preparation pipeline for machine learning classification. This project was developed as part of my undergraduate final thesis.

---

## 🧾 Table of Contents

1. [About](#about)  
2. [Dataset / Resources](#dataset--resources)  
3. [Dependencies](#dependencies)  
4. [Usage / Workflow](#usage--workflow)  
5. [Files / Structure](#files--structure)  
6. [Run Instructions](#run-instructions)  
7. [License](#license)  
8. [Contact](#contact)  

---

## About

This project implements preprocessing steps on photoplethysmography (PPG) signals aimed at stress detection. Tasks include:

- Filtering raw PPG data to remove noise/artifacts  
- Normalization / scaling  
- Signal segmentation / extraction (e.g. train / test split)  
- Feature engineering as needed (if any)  
- Preparation of data ready for classifier training and evaluation  

---

## Dataset / Resources

- Dataset: *Win-Ken Beh, Yi-Hsuan Wu, An-Yeu (Andy) Wu, “MAUS: A Dataset for Mental Workload Assessment on N-back task Using Wearable Sensor”, IEEE Dataport, May 10, 2021.* :contentReference[oaicite:0]{index=0}  
- Other resources / literature used (filter design, denoising methods, etc.)  

---

## Dependencies

Python versions & libraries used (sesuaikan kalau ada tambahan):

