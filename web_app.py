from flask import Flask, render_template, request
import string
import math
import nltk
import pymorphy2
import json

def punct_triggers(message):
    exclamation_marks = 0
    punctuation_marks = 0
    for symbol in message:
        if symbol == '!':
            exclamation_marks += 1
        if symbol in string.punctuation:
            punctuation_marks += 1
    if punctuation_marks == 0:
        return 1
    elif exclamation_marks / punctuation_marks >= 0.5:
        return 1.15 #надо уточнить вес
    else:
        return 1


def bayes_spam(the_list): # собственно наивный байесовский классификатор
    result = 0
    for word in the_list:
        nik = 0
        if word in spam_triggers.keys():
            nik = spam_triggers[word]
        spam_probability = (a + nik) / (a * spam_m + spam_nk)
        result += math.log10(spam_probability)
    return result


def bayes_not_spam(the_list):
    result = 0
    for word in the_list:
        nik = 0
        if word in not_spam_triggers.keys():
            nik = not_spam_triggers[word]
        not_spam_probability = (a + nik) / (a * not_spam_m + not_spam_nk)
        result += math.log10(not_spam_probability)
    return result

def addressing(text): #наличие обращения по имени в начале
    prob_thresh = 0.4
    address = False
    morph = pymorphy2.MorphAnalyzer()
    i = 0
    while i <= 7 and i <= len(nltk.word_tokenize(text)) - 1:
        word = nltk.word_tokenize(text)[i]
        for p in morph.parse(word):
            if 'Name' in p.tag and p.score >= prob_thresh:
                address = True
        i += 1
    if not address:
        return 1.1 #надо уточнить вес
    else:
        return 1


def greeting(message): #наличие приветствия в начале
    fg = False
    greetings = ['привет', 'здравствуйте', 'здравствуй', 'доброе утро', 'добрый день', 'добрый вечер',
                 'доброго времени суток', 'уважаемый', 'уважаемая', 'дорогой', 'дорогая', 'дорогие', 'уважаемые',
                 'приветствую']
    for elem in greetings:
        if elem in message[:8]:
            fg = True
            continue
    if not fg:
        return 1.05 #надо уточнить вес
    else:
        return 1

def spam(message):
    punct = punct_triggers(message)
    for elem in string.punctuation:  # очистить текст письма от пунктуации
        message = message.replace(elem, "")
    address = addressing(message)
    message = message.lower()
    greet = greeting(message)
    message = message.split()
    p_spam = bayes_spam(message) / punct / address / greet
    p_not_spam = bayes_not_spam(message)
    print(p_spam, p_not_spam)
    if p_spam > p_not_spam:
        res = 'это спам'
    else:
        res = 'это не спам'
    return res

with open('spam_triggers.json', encoding='utf-8') as f:
    spam_triggers = json.load(f)

with open('not_spam_triggers.json', encoding='utf-8') as f:
    not_spam_triggers = json.load(f)

a = 1
not_spam_m = 15600
not_spam_nk = 11650
spam_m = 40300
spam_nk = 30546

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def process():
    if request.values:
        text = request.values.get('text')
        message = text
        res = spam(message)
        return render_template('result.html', result=res)
    return 'Hmmm'

if __name__ == '__main__':
    app.run(debug=True)
