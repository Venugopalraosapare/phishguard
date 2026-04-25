import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re

# ── 1. LOAD THE DATA ──────────────────────────────────────────────────────────
# Option 1 dataset columns are: "Email Text" and "Email Type"
df = pd.read_csv('emails.csv', usecols=['Email Text', 'Email Type'])

# Rename so the rest of the code is clean and readable
df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})

print("=" * 55)
print("  PHISHING EMAIL DETECTOR — MODEL TRAINING")
print("=" * 55)
print(f"\nDataset loaded: {len(df)} emails total")
print("\nLabel breakdown:")
print(df['label'].value_counts())

# ── 2. CLEAN THE DATA ─────────────────────────────────────────────────────────
# Drop rows where text or label is missing
before = len(df)
df = df.dropna(subset=['text', 'label'])
dropped = before - len(df)
if dropped > 0:
    print(f"\nDropped {dropped} rows with missing values")

# Map labels to numbers: Phishing Email=1, Safe Email=0
df['label'] = df['label'].map({'Phishing Email': 1, 'Safe Email': 0})

# Drop any rows where label didn't match (catches typos in the dataset)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Lowercase everything so "FREE" and "free" are treated the same
df['text'] = df['text'].str.lower()

# Remove extra whitespace and newlines
df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

print(f"\nAfter cleaning: {len(df)} emails ready for training")

# ── 3. SPLIT INTO TRAINING AND TESTING SETS ───────────────────────────────────
# 80% used to train the model, 20% saved to test how accurate it is
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,      # 20% held back for testing
    random_state=42     # makes results repeatable every run
)

print(f"\nTraining set : {len(X_train)} emails")
print(f"Testing set  : {len(X_test)} emails")

# ── 4. CONVERT TEXT TO NUMBERS (TF-IDF) ──────────────────────────────────────
# Computers can't understand words directly.
# TF-IDF gives every word a score based on:
#   - how often it appears in THIS email (TF)
#   - how rare it is across ALL emails (IDF)
# So "verify-account" scores high (rare + meaningful)
# and "the" scores low (appears everywhere, tells us nothing)
print("\nConverting email text to numerical features (TF-IDF)...")

vectorizer = TfidfVectorizer(
    stop_words='english',   # ignore "the", "is", "and" etc.
    max_features=5000,      # use only the 5000 most useful words
    ngram_range=(1, 2),     # look at single words AND pairs like "click here"
    sublinear_tf=True       # smooths out very frequent terms
)

X_train_vec = vectorizer.fit_transform(X_train)  # learn vocabulary from training data
X_test_vec  = vectorizer.transform(X_test)        # apply same vocabulary to test data

print(f"Vocabulary size: {len(vectorizer.vocabulary_)} unique terms")

# ── 5. TRAIN THE MODEL ────────────────────────────────────────────────────────
# Naive Bayes works by learning:
#   "Given this word appears, what is the probability this is phishing?"
# It's fast, accurate for text, and great for a first ML project.
print("\nTraining Naive Bayes classifier...")

model = MultinomialNB(alpha=0.1)  # alpha=0.1 slightly improves accuracy over default
model.fit(X_train_vec, y_train)

print("Training complete!")

# ── 6. EVALUATE ACCURACY ──────────────────────────────────────────────────────
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print(f"\n{'=' * 55}")
print(f"  ACCURACY: {accuracy * 100:.1f}%")
print(f"{'=' * 55}")

# Full breakdown — useful for your README and interviews
print("\nDetailed performance report:")
print(classification_report(
    y_test,
    predictions,
    target_names=['Legitimate (Safe)', 'Phishing']
))

# ── 7. SHOW TOP PHISHING WORDS ────────────────────────────────────────────────
# This is a great thing to mention in your portfolio/interviews.
# It shows you understand what the model actually learned.
print("Top 15 words most associated with PHISHING:")
feature_names = vectorizer.get_feature_names_out()
class_index = 1  # 1 = phishing
top_indices = model.feature_log_prob_[class_index].argsort()[-15:][::-1]
top_words = [feature_names[i] for i in top_indices]
print(" | ".join(top_words))

print("\nTop 15 words most associated with LEGITIMATE email:")
class_index = 0  # 0 = legitimate
top_indices = model.feature_log_prob_[class_index].argsort()[-15:][::-1]
top_words = [feature_names[i] for i in top_indices]
print(" | ".join(top_words))

# ── 8. MANUAL TEST ────────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("  MANUAL TEST — sample predictions")
print(f"{'=' * 55}")

test_emails = [
    "URGENT: Your account has been suspended. Click here immediately to verify your identity or lose access.",
    "Hi Sarah, just following up on the Q3 report we discussed yesterday. Let me know your thoughts.",
    "Congratulations! You have been selected as a winner. Claim your $500 Amazon gift card before it expires!",
    "Team meeting moved to 3pm on Thursday. Please update your calendars and bring the project notes.",
    "Dear valued customer, your PayPal account is limited. Confirm your details within 24 hours to avoid suspension."
]

for email in test_emails:
    vec = vectorizer.transform([email.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "PHISHING  " if pred == 1 else "LEGITIMATE"
    confidence = round(max(prob) * 100, 1)
    print(f"[{label} — {confidence:5.1f}%]  {email[:65]}...")

# ── 9. SAVE THE MODEL ─────────────────────────────────────────────────────────
# pickle saves your trained model as a file.
# The web app will load these files instead of retraining every time.
pickle.dump(model,      open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print(f"\n{'=' * 55}")
print("  SAVED: model.pkl and vectorizer.pkl")
print("  Phase 2 complete! Move on to Phase 3.")
print(f"{'=' * 55}\n")