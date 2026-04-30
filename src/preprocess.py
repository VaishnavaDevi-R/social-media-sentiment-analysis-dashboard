import re
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", str(text))
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stopwords.words('english')])
    return text