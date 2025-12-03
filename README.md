# URL Phishing Detector 
A machine learning project that uses a fine-tuned DistilBERT model to classify URLs as **legitimate** or **phishing**.  
This project includes a full training pipeline and an interactive Streamlit web app for real-time URL classification.

---

## 🚀 Features
- Fine-tuned **DistilBERT** transformer model  
- **Binary**: `legitimate` or `phishing`
- **Streamlit web application** for real-time predictions
- ML pipeline with train/validation/test split  
- Model evaluation using **Accuracy** and **F1 Score**

---

## 📁 Project Structure

```
URL-Phishing-Detector/
│
├── data/
│   └── README.md          ← explains how to add the dataset (CSV not uploaded)
│
├── models/
│   └── README.md          ← explains where the fine-tuned model will be saved
│
├── src/
│   ├── training.py        ← model training + evaluation script
│   └── app.py             ← Streamlit app for URL classification
│
├── .gitignore
├── requirements.txt
└── README.md              ← this file
```


## Training the Model

Run the training script:

```bash
python src/training.py
```

This will:

- Load your dataset  
- Split into train/validation/test  
- Fine-tune DistilBERT  
- Evaluate model using accuracy and F1  
- Save your model inside:

```
models/fine_tuned_model/
```

---

## Running the Streamlit App

To launch the interactive web interface:

```bash
streamlit run src/app.py
```

Enter any URL and the model will classify it as:

- ✅ **LEGITIMATE**  
- 🚨 **PHISHING**

---

## Screenshots

Create a `/screenshots` folder in the repo root and upload your images there.

Recommended screenshots:
- Streamlit UI home screen  
- Classification example (legitimate)  
- Classification example (phishing)  

Then reference them here, for example:

```
![App Screenshot](screenshots/app_example.png)
```

---

## Technologies Used
- Python  
- Hugging Face Transformers  
- DistilBERT  
- PyTorch  
- Streamlit  
- scikit-learn  
- HuggingFace Datasets  
- Evaluate (Accuracy, F1)

---

## Notes
- Large dataset (`URL_dataset.csv`) is **NOT included** due to GitHub size limits.
- Fine-tuned model files are also **NOT included** for the same reason.
- Both are stored locally and created when running the training script.

---


