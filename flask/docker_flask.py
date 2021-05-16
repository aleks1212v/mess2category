from flask import Flask, json, request
#from flask_cors import CORS

import pandas as pd
import numpy as np
import re

import emoji
import pymorphy2
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import torch
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences

import artm

import os
import sqlite3
from datetime import datetime

currdir = os.getcwd()
app = Flask(__name__)

#загрузка BERT_pytorch
output_dir = os.path.join(currdir,'BERT_MODEL_PRETRAINED-предложения')
bert_tokenizer = BertTokenizer.from_pretrained(output_dir)
bert_model = BertForSequenceClassification.from_pretrained(output_dir)
bert_model.to('cpu')
bert_model.eval()

#загрузка BigARTM
artm_dir = os.path.join(currdir, 'artm_batches_categories')


def preprocessing_for_bert(text):
    def replace_three_or_more(text): #remove dublicate of symbol
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        return pattern.sub(r"\1\1", text)

    def emo_to_text(text):  
        emoticon = {'улыбка шутка' : ['\:\)', '\:\-\)', '\:\=\)'],
                    'грусть разочарование' : ['\:\(', '\:\-\(', '\:\=\('],
                    'открытая радость' : ['\=\)', '\=\-\)'],
                    'усмешка хихиканье' : ['\:>', '\:\->', '\:\=>'],
                    'улыбка' : ['\:\]', '\:\-\]', '\:\=\]'],
                    'смех' : ['\:D', '\:\-D', '\:\=D'],
                    'сильный смех' : ['\:DD', '\:\-DD', '\:\=DD'],
                    'сарказм' : ['\:\}', '\:\-\}', '\:\=\}'],
                    'подмигивание заигрывание' : ['\;\)', '\;\-\)', '\;\=\)'],
                    'молчание' : ['\:Х', '\:\-Х', '\:\=Х'],
                    'удивленное разочарование' : ['8\(', '8\-\(', '8\=\('],
                    'изумление' : ['B\-о', 'B\=о'],
                    'радостное удивление' : ['\%\)'],
                    'зевота' : ['\|\-o', '\|\=o'],
                    'смущение' : ['\:S', '\:-S', '\:\=S'],
                    'равнодушие скука' : ['\:\|', '\:\-\|', '\:\=\|'],
                    'недоверие сомнение' : ['\:\/', '\:\-\/', '\:\=\/'],
                    'дразнящее показывание языка' : ['\:ь', '\:\-ь', '\:\=ь'],
                    'поцелуй' : ['\:\*', '\:\^\*', '\:\-Ф'],
                    'шалость дурачество' : ['\:0\)'],
                    'кривая ухмылка' : ['\:7', '\:\-7']}

        for key, value in emoticon.items():
            text = re.sub(r"|".join(value), ' ' + key, text)
        return text

    try:
        text = emo_to_text(text)
        text = emoji.demojize(text, delimiters=("", "")) #преобразует эмодзи-символы в слова
        text = text.lower().replace("ё", "е")
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
        text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text) #удаляем цифры и спецсимволы
        text = replace_three_or_more(text) #удаляет дублирование букв, если число букв более 3
        text = re.sub(' +', ' ', text)
        return text.strip()
    except:
        #print(f'Слово {text} не прошло препроцессинг')
        return '' 

def bert_predict(mess_bert): #токенизация для берт и предсказание модели
    mess_bert = "[CLS] " + mess_bert + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(mess_bert)
    
    input_id = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    input_id = pad_sequences([input_id], maxlen=50, dtype="long", truncating="post", padding="post")
    attention_mask = [[float(i>0) for i in input_id[0]]]
    
    input_id = torch.tensor(input_id)
    attention_mask = torch.tensor(attention_mask)
    
    logits = bert_model(input_id, token_type_ids=None, attention_mask=attention_mask)
    cat_pred = torch.softmax(logits[0], dim = -1).tolist()[0]
    print(cat_pred)
    #cat_pred = cat_pred[0].tolist()[0]
    return cat_pred
    
def bert_top3(cat_pred):
    try:
        predict_values = []
        cat_path = os.path.join(currdir, r'flask/categories.txt')
        with open(cat_path, 'r') as f:
            for line in f:
                n, cat = [obj.strip('\n') for obj in line.split(';')]
                predict_values.append([n, cat, np.round(cat_pred[int(n)], 3)])
        estimates = pd.DataFrame(predict_values, columns = ['id', 'category', 'value'])
        print(estimates)
        top_val = pd.DataFrame(np.zeros((3,3)), columns = ['id', 'category', 'value'])
        for i in range(3):
            top_index = estimates['value'].idxmax()
            top_val.iloc[i] = estimates.iloc[top_index]
            estimates.loc[top_index, 'value'] = estimates['value'].min()
        top_val.reset_index(drop=True, inplace=True)
    except:
        print('Ошибка открытия файла')
    
    return top_val


    
"""________________________________________________________"""
#извлечение именованных сущностей

from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    LOC,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

def delete_NER(words):
    nf_words = ' '.join(words)
    per_words = []
    loc_words = []
    doc = Doc(nf_words)

    doc.segment(segmenter)

    doc.tag_ner(ner_tagger)

    for span in doc.spans:
        span.normalize(morph_vocab)
    for span in doc.spans:
        if span.type == 'PER':
            span.extract_fact(names_extractor)
            per_words.append(span.text)
        if span.type == 'LOC':
            span.extract_fact(names_extractor)
            loc_words.append(span.text)
       
    for word in per_words:
        if word in nf_words:
            nf_words = nf_words.replace(word, ' PER ')
    for word in loc_words:
        if word in nf_words:
            nf_words = nf_words.replace(word, ' LOC ')
    words = nf_words.split(' ')
    return words  
    
"""________________________________________________________"""


def preprocessing_for_bigartm(raw_text):
    #стемминг
    morph = pymorphy2.MorphAnalyzer() #по умолчанию русский язык
    stemmer = SnowballStemmer("english")
    stops = set(stopwords.words("english")) | set(stopwords.words("russian"))
    words = raw_text.split(' ')
    words = delete_NER(words) #удаление именованных сущностей: ФИО и гео-данных
    nf_words = list()
    for word in words:
        try:
            #word = word.encode('utf8mb4')
            ru_word = re.sub('[^а-яА-Я]+', ' ', word)
            en_word = re.sub('[^a-zA-Z]+', ' ', word)
            if ru_word == word and word != '': # кириллица
                parse_word = morph.parse(word)[0]
                nf_word = parse_word.normal_form
                nf_words.append(nf_word.strip())
            elif en_word == word and word != '': # латиница
                nf_word = stemmer.stem(word)
                nf_words.append(nf_word.strip())
            else:
                pass #смешанные слова из кириллицы и латиницы не анализируются
        except Exception as e:
                print(f'Слово не преобразовано {e}') 
                continue
    nf_words = [w for w in nf_words if not w in stops]
    nf_words = ' '.join(nf_words)  
    nf_words = nf_words.lower().replace("ё", "е")
    return nf_words

def bigartm_predict(mess_bigartm, top1_cat):
    try:
        mess_bigartm = ' |text ' + mess_bigartm

        #загрузка модели
        T = 10
        model_artm = artm.ARTM(num_topics = T, topic_names = ['sbj' + str(i) for i in range(T)], class_ids = {'text':1})
        model_artm.load(os.path.join(artm_dir, top1_cat + ".dump"))

        #сохранить текст в файл
        with open(os.path.join(currdir, 'flask/test_artm.txt'), 'w') as f:
            f.write(mess_bigartm + '\n')

        batch_vectorizer_test = artm.BatchVectorizer(data_path = os.path.join(currdir, 'flask/test_artm.txt'), data_format = 'vowpal_wabbit',
                                       target_folder = os.path.join(currdir, 'flask/test'), batch_size = 100)
        theta_test = model_artm.transform(batch_vectorizer=batch_vectorizer_test)
      
    except Exception:
        print('Ошибка загрузки bigartm')
    return theta_test

def bigartm_top3(cat_pred, top1_cat):
    try:
        cat_pred.columns = ['value']
        cat_pred['sbj'] = cat_pred.index
        cat_pred.reset_index(drop=True, inplace=True)

        top_val = pd.DataFrame(np.zeros((3,2)), columns = ['sbj', 'value'])
        for i in range(3):
            top_index = cat_pred['value'].idxmax()
            top_val.iloc[i] = cat_pred.iloc[top_index]
            cat_pred.loc[top_index, 'value'] = cat_pred['value'].min()
        top_val.reset_index(drop=True, inplace=True)

        top_words = pd.read_csv(os.path.join(artm_dir, top1_cat + ".top_words.csv"), sep = ';', index_col = 'index')
        words = []
        for key in top_val.index:
            text = ', '.join(map(lambda x: x.split(':')[0], top_words.loc[top_val.loc[key, 'sbj']].tolist()))
            words.append(text)
        top_val['top_words'] = words

        top_val = top_val[top_val['value'] > 0]
        top_val['value'] = np.round(top_val['value'], 3)
    except:
        print("Ошибка открытия")
    
    return top_val

def executor_top3(top1_cat, top1_sbj):
    #try:
    predict_values = []
    table_path = os.path.join(artm_dir, top1_cat + ".table.csv")
    worker_prob = np.round(pd.read_csv(table_path, sep = ';', index_col = 'index'), 2)

    sbj = top1_sbj
    top_1_arg = np.argmax(worker_prob.loc[sbj])
    top_1 = worker_prob.loc[sbj][top_1_arg]
    worker_prob.loc[sbj][top_1_arg] = 0
    top_2_arg = np.argmax(worker_prob.loc[sbj])
    top_2 = worker_prob.loc[sbj][top_2_arg]
    worker_prob.loc[sbj][top_2_arg] = 0
    top_3_arg = np.argmax(worker_prob.loc[sbj])
    top_3 = worker_prob.loc[sbj][top_3_arg]

    executor_dict = {worker_prob.columns[top_1_arg] : top_1, worker_prob.columns[top_2_arg] : top_2, worker_prob.columns[top_3_arg] : top_3}
    #except:
        #print('Ошибка открытия файла')
    
    return executor_dict


def check_username():
    try:
        log_path = os.path.join(currdir, r'flask/login.txt')
        username = request.args.get('username')
        password = request.args.get('password')
        with open(log_path, 'r') as f:
            for line in f:
                l, p = [obj.strip('\n') for obj in line.split(';')]
                if (l == username) and (p == password):
                    return True
    except Exception:
        print('Ошибка запроса')
    return False

def create_table_if_not_exists(conn, cursor):
    #cursor.execute("""DROP TABLE IF EXISTS mytable""")
    #conn.commit()
    cursor.execute("""CREATE TABLE IF NOT EXISTS mytable(
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(255) NOT NULL,
        message VARCHAR(1024) NOT NULL,
        category VARCHAR(1024) NOT NULL,
        subcategory VARCHAR(1024) NOT NULL,
        executor VARCHAR(1024) NOT NULL,
        topwords VARCHAR(512) NOT NULL,
        datetime DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        isvisible INTEGER DEFAULT 1)""")
    conn.commit()
    return
            
    
def store_message(username, message, cat_dict, subcat_dict, executor_dict, topwords):
    try:
        store_list = [username, message, json.dumps(cat_dict), json.dumps(subcat_dict), json.dumps(executor_dict), topwords]
        
        #подключение базы данных
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            
            query = 'SELECT 1 FROM mytable WHERE message = "' + message + '"'
            cursor.execute(query)
            conn.commit()
            rows = cursor.fetchall()
            print(rows)
            
            if len(rows) == 0: #запроса не было в базе, заносим его
                query = '''insert into mytable (username, message, category, subcategory, executor, topwords)
                           values (?, ?, ?, ?, ?, ?)'''
                conn.execute(query, store_list)
                conn.commit()
            else: #делаем данные видимыми
                query = 'UPDATE mytable SET isvisible = "1" WHERE username = ?'
                conn.execute(query, [username])
                conn.commit()

    except sqlite3.IntegrityError as e:
        print('Error occured: ', e)
    
    finally:
        if conn:
            conn.close()
    return True



# Cross Origin Resource Sharing (CORS) handling
#CORS(app, resources={'/image': {"origins": "http://localhost:8080"}})

@app.route('/message', methods=['POST']) #загрузка новых обращений граждан в систему и выдача результата
def message_post_request(): 
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    try:

        user = request.args.get('username')

        body = json.loads(request.data.decode('utf-8')) #преобразование байтов в строку
        message = body['mess']

        #запуск bert
        mess_bert = preprocessing_for_bert(message)

        predict_values = bert_predict(mess_bert)
        top3_values = bert_top3(predict_values)
        top1_category = top3_values.loc[top3_values['value'].idxmax(), 'category']
        cat_dict = top3_values.to_dict()

        #запуск bigartm
        mess_bigartm = preprocessing_for_bigartm(mess_bert)

        predict_values_artm = bigartm_predict(mess_bigartm, top1_category)
        top3_artm = bigartm_top3(predict_values_artm, top1_category)
        top_words = top3_artm.loc[top3_artm['value'].idxmax(), 'top_words']
        top1_sbj = top3_artm.loc[top3_artm['value'].idxmax(), 'sbj']
        subcat_dict = top3_artm.to_dict()
        
        #вывод исполнителей
        executor_dict = executor_top3(top1_category, top1_sbj)
        
        store_message(user, message, cat_dict, subcat_dict, executor_dict, top_words) #сохранить сообщение в базу данных

    except Exception:
        print('Ошибка обработки обращения')
        return json.dumps({'result': 'ERROR'})
    return json.dumps({'result': 'SUCCESS', 'top1_category': top1_category, 'category': cat_dict, 'subcategory': subcat_dict, \
                       'executor': executor_dict, 'top_words': top_words, 'message':body}).encode('utf-8')

def get_message(username, count):
    #подключение базы данных
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            if username != 'admin':
                query = 'SELECT * FROM mytable WHERE username = ? AND isvisible = "1" ORDER BY datetime DESC LIMIT ?'
                param_list = [username, count]
            else:
                query = 'SELECT * FROM mytable ORDER BY datetime DESC LIMIT ?'
                param_list = [count]
            cursor.execute(query, param_list)
            conn.commit()
            rows = cursor.fetchall()
            print(rows)
    except Exception:
        print('Ошибка обработки')
    return rows

@app.route('/message', methods=['GET'])
def message_get_request():  #выводить count последних запросов данного пользователя, если они не скрыты
    #если пользователь Админ, то выводить count последних запросов всех пользователей, независимо от того, скрыты они или нет
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    count = request.args.get('count')
    
    mess_list = get_message(username, count)
    
    return json.dumps({'result':mess_list}).encode('utf-8')

def delete_message(username):
    #пользователь может удалить свои сообщения (isvisible = 0), но администратор все равно сможет их просматривать.
    #администратор может окончательно удалить все сообщения.
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            if username != 'admin':
                query = 'UPDATE mytable SET isvisible = "0" WHERE username = ?'
                param_list = [username]
            else:
                query = 'DELETE FROM mytable'
                param_list = []
            cursor.execute(query, param_list)
            conn.commit()
    except Exception:
        print('Ошибка обработки')
    return 

@app.route('/message', methods=['DELETE'])
def message_delete_request():  
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    
    delete_message(username)
    
    return json.dumps({'result': 'SUCCESS'}).encode('utf-8')

def get_statistics(username):
    #пользователь получает статистику о себе, а админ обо всех
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            query = 'SELECT DISTINCT username FROM mytable'
            cursor.execute(query)
            conn.commit()
            users = cursor.fetchall()
            stat_dict = {}
            if username == 'admin':
                for user in users:
                    query = 'SELECT COUNT(*) FROM mytable WHERE username = ?'
                    param_list = [user[0]]
                    cursor.execute(query, param_list)
                    conn.commit()
                    count = cursor.fetchall()[0][0]
                    stat_dict.update({user[0] : count})
            else:
                query = 'SELECT COUNT(*) FROM mytable WHERE username = ?'
                param_list = [username]
                cursor.execute(query, param_list)
                conn.commit()
                count = cursor.fetchall()[0][0]
                stat_dict.update({username : count})
    except Exception:
        print('Ошибка обработки')
    return stat_dict

@app.route('/statistics', methods=['GET'])
def statistics_get_request():  #сколько сообщений загружено данным пользователем, если ты админ, то всеми
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    
    stat_dict = get_statistics(username)
    
    return json.dumps({'result': stat_dict}).encode('utf-8')

def get_categories(username):
    #пользователь получает категорию последнего запроса
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            if username != 'admin':
                query = 'SELECT message, category FROM mytable WHERE username = ? AND isvisible = "1" ORDER BY datetime DESC LIMIT 1'
                param_list = [username]
                cursor.execute(query, param_list)
            else:
                query = 'SELECT message, category FROM mytable ORDER BY datetime DESC LIMIT 1'
                cursor.execute(query)
            conn.commit()
            message, category = cursor.fetchall()[0]
        
    except Exception:
        print('Ошибка обработки')
        return (0,0)
    return  (message, category)


@app.route('/categories', methods=['GET'])
def categories_get_request():  
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    
    (message, category) = get_categories(username)
    if category == 0:
        return json.dumps({'result': 'ERROR'})
    
    return json.dumps({'result': 'SUCCESS', 'message': message, 'category':category}).encode('utf-8')


def get_topwords(username):
    #пользователь получает список топ-слов последнего запроса
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            if username != 'admin':
                query = 'SELECT message, topwords FROM mytable WHERE username = ? AND isvisible = "1" ORDER BY datetime DESC LIMIT 1'
                param_list = [username]
                cursor.execute(query, param_list)
            else:
                query = 'SELECT message, topwords FROM mytable ORDER BY datetime DESC LIMIT 1'
                cursor.execute(query)
            conn.commit()
            message, topwords = cursor.fetchall()[0]
        
    except Exception:
        print('Ошибка обработки')
        return (0,0)
    return  (message, topwords)

@app.route('/topwords', methods=['GET'])
def topwords_get_request():  
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    
    (message, topwords) = get_topwords(username)
    if topwords == 0:
        return json.dumps({'result': 'ERROR'})
    return json.dumps({'result': 'SUCCESS', 'message': message, 'topwords':topwords}).encode('utf-8')


def get_executor(username):
    #пользователь получает список топ-слов последнего запроса
    try:
        conn = sqlite3.connect(os.path.join(currdir, r"flask/log.db")) # или :memory: чтобы сохранить в RAM
        with conn:
            cursor = conn.cursor()
            create_table_if_not_exists(conn, cursor)
            if username != 'admin':
                query = 'SELECT message, executor FROM mytable WHERE username = ? AND isvisible = "1" ORDER BY datetime DESC LIMIT 1'
                param_list = [username]
                cursor.execute(query, param_list)
            else:
                query = 'SELECT message, executor FROM mytable ORDER BY datetime DESC LIMIT 1'
                cursor.execute(query)
            conn.commit()
            message, executor = cursor.fetchall()[0]
        
    except Exception:
        print('Ошибка обработки')
        return (0,0)
    return  (message, executor)

@app.route('/executor', methods=['GET'])
def executor_get_request():  
    check = check_username()
    if not check:
        return json.dumps({'result':'Некорректный логин или пароль!'}).encode('utf-8')
    
    username = request.args.get('username')
    
    (message, executor) = get_executor(username)
    if executor == 0:
        return json.dumps({'result': 'ERROR'})
    
    return json.dumps({'result': 'SUCCESS', 'message': message, 'executor':executor}).encode('utf-8')



if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000)
 