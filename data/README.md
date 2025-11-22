# Data Folder

This folder stores the dataset required for training the URL Phishing Detector model.

The file `URL_dataset.csv` is not included in this repository because it is too large for GitHub.

To use this project, download or provide your own dataset and place it in this folder with the exact name:

data/URL_dataset.csv

The dataset must contain the following columns:
- url — contains the URL of the website
- type — specifies whether the URL is legitimate or phishing

Example:

url,type
http://example.com,legitimate
http://malicious-url.biz,phishing

Notes:
- The dataset stays local on your computer.
- The training script (src/training.py) automatically loads the dataset from data/URL_dataset.csv.
