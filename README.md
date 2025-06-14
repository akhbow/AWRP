# Asymmetric Windowing Recurrence Plots (AWRP) for EEG-Based Emotion Recognition

This repository contains the official implementation of the AWRP encoding process from the paper:

**"Enhancing EEG-Based Emotion Recognition Using Asymmetric Windowing Recurrence Plots"**  
*Published in IEEE Access, 2024*  
DOI: [10.1109/ACCESS.2024.3409384](https://doi.org/10.1109/ACCESS.2024.3409384)

## 📁 Contents

- `AWRP_encoding_DEAP.py`: AWRP generator for the DEAP dataset (32 subjects, 40 trials each, 32 channels).
- `AWRP_encoding_SEED_sub1.py`: AWRP generator for subject 1 of the SEED dataset (15 trials, 64 channels).

## 🧠 Method Overview

This implementation converts EEG signals into Asymmetric Windowing Recurrence Plots (AWRPs) by:
1. Segmenting EEG signals into fixed-length windows.
2. Generating recurrence plots (RPs) from each segment.
3. Combining adjacent RPs into averaged RPs (ARPs).
4. Arranging ARPs into a 2D image grid to form the final AWRP.
5. Saving AWRPs as `.png` images per subject-trial-channel.

## ⚙️ Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- OpenCV (`cv2`)
- pyts
- Pillow
- psutil

Install with:

```bash
pip install numpy scipy matplotlib opencv-python pyts pillow psutil
```

## 🚀 How to Run

For DEAP dataset (32 subjects):
```bash
python AWRP_encoding_DEAP.py
```

For SEED dataset (subject 1 only):
```bash
python AWRP_encoding_SEED_sub1.py
```

> 🔧 **Note**: The SEED script is tailored for subject 1. To process other subjects, trial key names (`djc_eegX`) must be updated accordingly based on the SEED dataset's structure.

## 📂 Output

- AWRPs are saved as `.png` images in folders:
  - `AWRP_DEAP_8/`
  - `AWRP_SEED_8/`
- Timing and memory usage logs are written to:
  - `avg_time-memory_usage_AWRP_8.txt`

## 📄 Citation

If you use this code, please cite our paper:

```bibtex
@article{prabowo2024,
  author    = {Wahyu Prabowo, Dwi and Akhmad Setiawan, Noor and Debayle, Johan and Nugroho, Hanung Adi},
  journal   = {IEEE Access},
  title     = {Enhancing EEG-Based Emotion Recognition Using Asymmetric Windowing Recurrence Plots},
  year      = {2024},
  volume    = {12},
  doi       = {10.1109/ACCESS.2024.3409384}
}
```

## 🧾 License

This code is for academic use only. For other purposes, please contact the authors.
