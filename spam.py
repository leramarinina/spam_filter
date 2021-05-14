from nltk.tokenize import wordpunct_tokenize
from pymorphy2 import MorphAnalyzer
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import mailbox, re
from bs4 import BeautifulSoup
import json
import os


def get_top_tf_idf_words(tfidf_vector, feature_names, top_n):
    sorted_nzs = np.argsort(tfidf_vector.data)[:-(top_n+1):-1]
    keys = feature_names[tfidf_vector.indices[sorted_nzs]]
    values = tfidf_vector.indices[sorted_nzs]
    dct = {}
    for k in range(len(keys)):
        dct[keys[k]] = values[k]
    return dct

mbox = mailbox.mbox(r"Спам.mbox")

for i, message in enumerate(mbox):
    content = message.get_payload(decode=True)
    if content:
        text = content.decode("utf-8", "replace")
        soup = BeautifulSoup(text, 'html.parser')
        item = soup.span
        if item:
            item = item.contents
            for elem in item:
                elem = str(elem).strip()
                elem_clear = re.sub(r'<[^<]+>', '', elem)
                with open('spam_file.txt', 'a', encoding='utf-8') as new_file:
                    new_file.write(elem_clear)

path_to_articles = 'C:\Lessons\проект'
articles = [os.path.join(path_to_articles, item)
    for item in os.listdir(path_to_articles)
    if item.endswith('.txt') and item != 'not_spam_file.txt']

articles_texts = []

for article_path in articles:
    with open(article_path, encoding='utf8') as a_src:
        articles_texts.append("\n".join(a_src.readlines()))

morph = MorphAnalyzer()

spam_m = 0 #количество слов в обучающей выборке
spam_nk = 0 #количество слов в обучающей выборке без стоп-слов
stops = stopwords.words("russian")
articles_preprocessed = []
for a_text in articles_texts:
    a_tokens = wordpunct_tokenize(a_text)
    a_lemmatized = ' '.join([morph.parse(item)[0].normal_form for item in a_tokens])
    articles_preprocessed.append(a_lemmatized)
    for token in a_tokens:
        p = morph.parse(token)[0]
        if p.tag.POS:
            spam_m += 1
        if p.tag.POS and token not in stops:
            spam_nk += 1

tfidf = TfidfVectorizer(analyzer="word", stop_words=stops)

articles_tfidf = tfidf.fit_transform(articles_preprocessed)

feature_names = np.array(tfidf.get_feature_names())

spam_triggers = {}

for i, article in enumerate(articles_texts):
    article_vector = articles_tfidf[i, :]
    if article[:6] == '1 445 ':
        words = get_top_tf_idf_words(article_vector, feature_names, 200)
    else:
        words = get_top_tf_idf_words(article_vector, feature_names, 5)
    if words:
        for key in words:
            t = morph.parse(key)[0]
            if 'Name' in t.tag and t.score >= 0.4:
                continue
            if key not in spam_triggers.keys():
                spam_triggers[key] = int(words[key])
            else:
                spam_triggers[key] += int(words[key])

with open('spam_triggers.json', 'w', encoding='utf-8') as file:
    json.dump(spam_triggers, file)
print(spam_m, spam_nk)