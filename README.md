# Tourism Package Prediction

This project builds and deploys a Machine Learning pipeline to predict whether a customer will purchase a tourism package, using **MLOps practices** with Hugging Face Hub and GitHub Actions.

---

## Project Overview
- **Dataset**: [tourism-package-prediction-dataset](https://huggingface.co/datasets/moulibasha/tourism-package-prediction-dataset)  
- **Preprocessed Splits**: [tourism-package-prediction-train-test](https://huggingface.co/datasets/moulibasha/tourism-package-prediction-train-test)  
- **Model**: [tourism-package-prediction-model](https://huggingface.co/moulibasha/tourism-package-prediction-model)  
- **Deployment (Streamlit Space)**: [tourism-package-prediction](https://huggingface.co/spaces/moulibasha/tourism-package-prediction)

---

## Repository Structure
```
├── .github/
│ └── workflows/
│ └── pipeline.yml # CI/CD pipeline (GitHub Actions)
├── ci_requirements.txt # Requirements for pipeline jobs
├── tourism_project/
│ ├── data/ # Raw dataset (tourism.csv)
│ ├── model_building/ # Data registration & training scripts
│ └── deployment/ # Deployment files
│ ├── app.py # Streamlit app
│ ├── Dockerfile # Container configuration
│ ├── requirements.txt # Dependencies for deployment
│ └── push_space.py # Script to push app to Hugging Face Space
└── README.md
```
---

## Workflow
1. **Data Registration**: Upload raw CSV to Hugging Face dataset repo  
2. **Data Preparation**: Clean, split, and upload train/test datasets  
3. **Model Training**: Tune Random Forest model, log with MLflow, push to Hugging Face Model Hub  
4. **Deployment**: Streamlit app & Dockerfile pushed to Hugging Face Space  
5. **Automation**: GitHub Actions orchestrates the entire workflow end-to-end  

---

## Tech Stack
- Python, Scikit-learn, Pandas, Numpy  
- Hugging Face Hub (Datasets, Model Hub, Spaces)  
- Streamlit for deployment UI  
- GitHub Actions for CI/CD  
- MLflow for experiment tracking  

---

## How to Run Locally
```bash
# clone the repo
git clone https://github.com/mouli-basha/tourism-package-prediction.git
cd tourism-package-prediction/tourism_project/deployment

# install requirements
pip install -r requirements.txt

# run the app
streamlit run app.py

