import duckdb 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import  classification_report
from src.preprocessing.text_cleaner import clean_text


conn = duckdb.connect('catalog.db')

train_data = conn.execute("""
    SELECT title, category_id 
    FROM products 
    WHERE split = 'train' AND title IS NOT NULL AND title != '' AND category_id IS NOT NULL
""").fetchall()
train_text,train_labels = zip(*train_data)
cleaned_train_text = list(map(clean_text, train_text))

test_data = conn.execute("""
    SELECT title, category_id 
    FROM products 
    WHERE split = 'test' AND title IS NOT NULL AND title != '' AND category_id IS NOT NULL
""").fetchall()
test_text,test_labels = zip(*test_data)
cleaned_test_text = list(map(clean_text, test_text))

tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(cleaned_train_text)
X_test_tfidf = tfidf_vectorizer.transform(cleaned_test_text)
print(f"TF-IDF training data shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing data shape: {X_test_tfidf.shape}")

clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, train_labels)
y_pred = clf.predict(X_test_tfidf)
CR = classification_report(test_labels, y_pred)
print(f"Classification Report: {CR}")

