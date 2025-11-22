# Models Folder

This folder is used to store the fine-tuned model for the URL Phishing Detector.

The actual model files are not included in this repository because they are too large for GitHub.

After training the model using src/training.py, a folder will be created here:

models/fine_tuned_model/

Inside that folder you should expect files such as:
- config.json
- pytorch_model.bin
- tokenizer.json
- tokenizer_config.json
- vocab.txt

Notes:
- The Streamlit app (src/app.py) loads the model from this folder.
