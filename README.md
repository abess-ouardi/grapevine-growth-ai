# Grapevine Growth Prediction Using GAN + Process-Based Modeling

**Hybrid AI model combining Process-Based Crop Models (PBCMs) and Pix2Pix GANs**  
Predicting structural grapevine development over time, as part of my Bachelor’s thesis in Automation Engineering – University of Bologna.

---

## Project Overview

This project implements a hybrid framework that merges **physiological modeling** (photosynthesis, biomass, phenology) with **deep generative vision** to simulate plant morphology under different environmental and pruning conditions.

The architecture consists of:
- A **PBCM** to simulate daily physiological growth (radiation use efficiency, GDD, etc.)
- A **Pix2Pix GAN** with U-Net generator and PatchGAN discriminator to reconstruct plant structure

---

## Key Technologies

- `Python`, `PyTorch`, `OpenCV`, `CUDA`
- RunPod cloud GPU backend
- GAN losses: **Adversarial**, **Dice**, **SSIM**, **Edge**

---

## Results

| Metric       | Train     | Test      |
|--------------|-----------|-----------|
| SSIM Score   | 0.92      | 0.81 ± 0.07 |
| Dice Score   | 0.87      | 0.74 ± 0.05 |

🔬 Training used 98 images from the [Buds Dataset](https://www.kaggle.com/datasets/frednavruzov/grapevine-buds-dataset), augmented and binarized to simulate pruning.

---

## Repository Structure

```
grapevine-growth-ai/
├── code/             ← model definitions, training, prediction
├── data/             ← CO2, soil, temperature, light input CSVs
├── models/           ← pre-trained GAN weight (Pix2Pix)
├── results/          ← test predictions vs real data
├── docs/             ← thesis presentation and optional summary
├── requirements.txt  ← Python dependencies
├── LICENSE
└── README.md
```

---

## Visuals

For a full overview of the hybrid modeling pipeline, architecture diagrams, and sample GAN outputs, see the PDF presentation:

👉 [presentation.pdf](docs/presentation.pdf)
---

## Future Work

- Expand dataset size and diversity
- Integrate real-time IoT sensing (for biomass inputs)
- Apply to other crops or natural vegetation
- Field validation and closed-loop integration with UGVs

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Abess Ouardi**  
Graduate in Automation Engineering, University of Bologna  
GitHub: [@abess-ouardi](https://github.com/abess-ouardi)

---

## Contributions / Feedback

Feel free to fork, clone, or open an issue.  
Feedback is welcome as I continue into my Master’s and future AI research!
