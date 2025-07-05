# OCR Arabic Handwritten

## Project Overview
This project is an end-to-end, interactive OCR system for Arabic handwritten text. It supports two core recognition engines:
- **CNN‑BLSTM‑CTC**: Built from scratch in PyTorch.
- **Microsoft TrOCR**: HuggingFace transformer model, fine-tuned on your data.

Users can upload images, view predictions, manually correct them, and save corrections for feedback learning. Performance is evaluated via Character Error Rate (CER) and Word Error Rate (WER).

---

## Folder Structure
```
ocr-arabic-handwritten/
│
├── notebooks/
│   └── ocr_pipeline_colab.ipynb    # Main Colab/Tutorial notebook
│
├── project/
│   ├── data/
│   │   ├── loaders.py             # load_dataset_from_csv, config reader
│   │   ├── preprocess.py          # transforms, augmentations
│   │   └── config.yaml            # dataset paths: khatt, ifn_enit, ahw, custom
│   │
│   ├── models/
│   │   ├── cnn_ctc.py             # CNN‐BLSTM‐CTC architecture
│   │   └── trocr.py               # TrOCR fine‐tuning wrapper
│   │
│   ├── training/
│   │   ├── train_ctc.py           # training loop, CTC collate
│   │   └── train_trocr.py         # fine‑tuning script
│   │
│   ├── evaluation/
│   │   └── metrics.py             # CER, WER implementations
│   │
│   ├── app/
│   │   ├── interface.py           # Streamlit/Gradio UI code
│   │   └── feedback.py            # saving corrections & feedback loop
│   │
│   ├── utils/
│   │   └── helpers.py             # common utilities (logging, save/load)
│   │
│   └── requirements.txt           # PyTorch, transformers, streamlit/gradio, etc.
│
├── data/                          # placeholder dirs for raw datasets
│   ├── khatt/
│   ├── ifn_enit/
│   ├── ahw/                       # optional
│   └── custom_data/
│
└── README.md                      # project overview, setup instructions
```

---

## Setup & Usage

### 1. Google Colab/Kaggle
- Open `notebooks/ocr_pipeline_colab.ipynb` in Colab or Kaggle.
- Mount Google Drive if using Colab.
- Install dependencies from `project/requirements.txt`.

### 2. Configuration
- Edit `project/data/config.yaml` to set dataset paths.
- Place your datasets in the `data/` directory or update paths accordingly.

### 3. Running the Pipeline
- Follow notebook sections: data loading, model training, evaluation, and interactive demo.
- Upload images, view/correct predictions, and save corrections.
- Corrections are saved as CSV for feedback learning.

### 4. Feedback Learning
- Corrections are appended to a CSV and can be integrated into the training set.
- Retrain models using updated data for improved accuracy.

### 5. Evaluation
- CER and WER are computed for each dataset.
- Results are displayed in tables/plots for easy comparison.

---

## Deliverables
- **Google Colab notebook**: Fully runnable, step-by-step.
- **Python package**: Modular code in `project/`.
- **config.yaml**: Dataset paths and hyperparameters.
- **Sample corrections CSV**: For feedback learning.
- **Demo video**: Short walkthrough of the system.
- **Support notes**: Guidance for post-delivery support.

---

## Adding New Data
- Add your images and labels to `data/custom_data/`.
- Update `config.yaml` with the new dataset path.
- Follow notebook instructions to include your data in training/evaluation.

---

## Extending the Project
- Add new model architectures in `project/models/`.
- Implement new evaluation metrics in `project/evaluation/`.
- Expand the UI in `project/app/interface.py`.

---

## CER/WER Targets
- Aim for CER < 0.90 for CRNN (CNN-BLSTM-CTC) on challenging datasets.
- Compare both models side-by-side in the notebook.

---

## Support
- For issues, consult the notebook and code comments.
- Post-delivery support notes are included in the repo. 