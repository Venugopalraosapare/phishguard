from flask import Flask, render_template, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load the saved model and vectorizer at startup
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model      = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

def clean_text(text):
    """Apply the same cleaning used during training."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_email(text):
    """Run the email through the model and return a result dict."""
    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    pred      = model.predict(vec)[0]
    proba     = model.predict_proba(vec)[0]
    confidence = round(float(max(proba)) * 100, 1)

    # Pull the top words that influenced this decision
    feature_names = vectorizer.get_feature_names_out()
    email_vec     = vec.toarray()[0]
    nonzero_idx   = email_vec.nonzero()[0]

    class_idx   = int(pred)  # 1=phishing, 0=legit
    log_probs   = model.feature_log_prob_[class_idx]
    scored      = [(feature_names[i], log_probs[i]) for i in nonzero_idx]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_words   = [w for w, _ in scored[:6]]

    return {
        'verdict':    'PHISHING' if pred == 1 else 'LEGITIMATE',
        'confidence': confidence,
        'is_phishing': bool(pred == 1),
        'top_words':  top_words,
        'phishing_pct': round(float(proba[1]) * 100, 1),
        'legit_pct':    round(float(proba[0]) * 100, 1),
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    email_text = data.get('email_text', '').strip()

    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    if len(email_text) < 10:
        return jsonify({'error': 'Email text too short'}), 400

    result = analyze_email(email_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
