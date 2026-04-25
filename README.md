# PhishGuard — Phishing Email Detector

> ML-powered web app that classifies emails as phishing or legitimate in real time.

**🔴 Live Demo:** [https://phishguard-4o3l.onrender.com](https://phishguard-4o3l.onrender.com)

---

## What it does

PhishGuard analyzes email text and uses a trained machine learning model to determine whether an email is a **phishing attempt** or **legitimate**. It returns a verdict, a confidence score, probability breakdown, and the top trigger words that influenced the decision.

---

## How it works

1. Email text is cleaned and lowercased
2. A **TF-IDF vectorizer** converts words into numerical features (unigrams + bigrams)
3. A **Naive Bayes classifier** scores the email against patterns learned from 18,650 labeled emails
4. The app returns a verdict with confidence percentage and the top words that drove the prediction

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 97%+ |
| Precision (Phishing) | 0.97 |
| Recall (Phishing) | 0.98 |
| F1 Score | 0.97 |
| Training data | 18,650 emails |

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (Naive Bayes, TF-IDF) |
| Web framework | Flask |
| Production server | Gunicorn |
| Hosting | Render (free tier) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project structure

```
phishguard/
├── app.py              # Flask web server + prediction logic
├── train_model.py      # ML training script
├── model.pkl           # Trained Naive Bayes classifier
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
└── templates/
    └── index.html      # Web UI
```

---

## Run locally

```bash
git clone https://github.com/Venugopalraosapare/phishguard
cd phishguard
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5000](http://localhost:5000)

---

## Retrain the model

If you want to retrain on new data:

```bash
# Place your dataset as emails.csv with columns: "Email Text", "Email Type"
python train_model.py
```

Labels should be `Phishing Email` and `Safe Email`. The script will output accuracy metrics and save new `model.pkl` and `vectorizer.pkl` files.

---

## What I learned

- How to build and evaluate a text classification pipeline with scikit-learn
- How TF-IDF vectorization works and why it's effective for email analysis
- How to wrap an ML model in a Flask web app with a REST API
- How to deploy a Python web service to production using Gunicorn and Render
- How phishing emails differ linguistically from legitimate ones (urgency, fear, reward language)

---

## Note on free tier

This app is hosted on Render's free tier. If the service hasn't been visited recently it may take **30–50 seconds** to wake up on the first request. This is normal behavior for free-tier hosting.

---

*Built as a portfolio project demonstrating ML, Python web development, and cybersecurity concepts.*
