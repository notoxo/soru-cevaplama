#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import spacy

# Bing Search API Configuration
api_key = '6199cb40be464309a664a70d549179bf'  # Replace with your actual API key
query = 'borsa'
headers = {'Ocp-Apim-Subscription-Key': api_key}
params = {'q': query, 'count': 50, 'offset': 0, 'mkt': 'tr-TR', 'safesearch': 'Moderate'}
url = 'https://api.bing.microsoft.com/v7.0/search'

# Making API Request
response = requests.get(url, headers=headers, params=params)
if response.status_code != 200:
    print(f"Request failed with status code {response.status_code}: {response.text}")
else:
    results = response.json()
    urls = [result['url'] for result in results.get('webPages', {}).get('value', [])]

# Additional URLs
additional_urls = [
    'https://tr.wikipedia.org/wiki/Borsa',
    'https://www.getmidas.com/borsa-terimleri/borsa-istanbul-nedir/#:~:text=Borsa%20İstanbul%3B%20yatırımcıları%20bir%20araya,borsa%20organizasyonu%20olma%20özelliği%20taşır.',
    'https://www.getmidas.com/midas-akademi/borsa-nedir/',
    'https://www.getmidas.com/midas-akademi/parca-hisse-senedi-nedir/',
    'https://www.getmidas.com/borsa-terimleri/gunluk-islem-nedir/',
    'https://eksiseyler.com/borsaya-yeni-baslayacaklara-rehber-olacak-bir-yazi',
    'https://www.kuveytturk.com.tr/blog/finans/yeni-baslayanlar-icin-borsa-kilavuzu'
]
urls.extend(additional_urls)

def extract_text_bs4(url, depth=1):
    if depth < 0:
        return []
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve data from {url}: {str(e)}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join(para.get_text() for para in paragraphs)
    
    # Extract URLs from anchor tags
    subpage_urls = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')]
    
    # Extract text from subpages
    subpage_texts = []
    for subpage_url in subpage_urls:
        subpage_texts.extend(extract_text_bs4(subpage_url, depth-1))
    
    return [text] + subpage_texts

# Extracting Text Data from URLs
all_text_data = []
for url in urls:
    all_text_data.extend(extract_text_bs4(url, depth=1))
merged_text_data = " ".join(all_text_data)

# Preprocessing
nltk.download('punkt')
cleaned_text = re.sub(r'\W', ' ', str(merged_text_data))
cleaned_text = re.sub(r'\s+', ' ', cleaned_text, flags=re.I)
cleaned_text = cleaned_text.lower().strip()
tokens = word_tokenize(cleaned_text)

# Stop-word Removal
stopwords_file_path = 'path_to_your_stopwords_file.txt'  # Update the path
with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    turkish_stopwords = [line.strip() for line in file]
filtered_tokens = [token for token in tokens if token not in turkish_stopwords]
preprocessed_text = ' '.join(filtered_tokens)

# Spacy Usage and Question Extraction
nlp = spacy.load('tr_core_news_trf')
doc = nlp(merged_text_data)
labels = []
texts = []
question_pattern = re.compile(r'(?i)\b(nasıl|ne|neden|ne zaman|kim|hangi|kaç)[\w\s]*\?')
for sent in doc.sents:
    texts.append(sent.text)
    labels.append("question" if question_pattern.search(sent.text) else "not_question")

