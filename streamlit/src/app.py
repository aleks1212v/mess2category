import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os

import requests
import json

import altair as alt

serverhost = 'localhost'#'message2category'
serverport = '5000'

currdir = os.getcwd()

#def load_username():
#    try:
#        log_path = os.path.join(currdir, r'login.txt')
#        logins = []
#        with open(log_path, 'r') as f:
#            for line in f:
#                l = line.strip('\n')
#                logins.append(l)
#        return logins
#    except Exception:
#        print('Ошибка запроса')
#    return ''

def write_content(raw_json):
    top1_category = raw_json['top1_category']
    category = pd.DataFrame(raw_json['category'])
    subcategory = pd.DataFrame(raw_json['subcategory'])
    executor = pd.DataFrame([raw_json['executor']])
    top_words = raw_json['top_words'].split(', ')
    message = ''.join(raw_json['message'].values())
    st.markdown('### Ваше обращение:')
    st.write(message)
    st.write('----------------')
    st.markdown('### Наиболее вероятная категория обращения:')
    st.write(top1_category)
    st.write('----------------')
    st.markdown('### Топ-3 наиболее вероятных категорий:')
    st.write(category)
    
    chart = alt.Chart(category).mark_bar(size=30).encode(x='value', y='category', color='value').properties(width=700, height = 300)
    st.altair_chart(chart)
    st.write('----------------')
    st.markdown('### Топ-3 из 10 подкатегорий:')
    st.write(subcategory)
    
    chart_2 = alt.Chart(subcategory).mark_bar(size=30).encode(x='value', y='top_words', color='value').properties(width=700, height = 300)
    st.altair_chart(chart_2)
    st.write('----------------')
    st.markdown('### Топ-3 наиболее вероятных исполнителей:')
    st.write(executor)
    
    ex = executor.T
    ex.columns = ['value']
    ex['name'] = ex.index
    ex.reset_index(drop=True, inplace=True)
    chart_3 = alt.Chart(ex).mark_bar(size=30).encode(x='value', y='name', color='value').properties(width=700, height = 300)
    st.altair_chart(chart_3)
    st.write('----------------')
    st.markdown('### Ключевые слова наиболее вероятной подкатегории:')
    st.write(top_words)
    return

def write_contents(result_frame):
    for record_number in range(result_frame.shape[0]): 
        index = result_frame.loc[record_number]['index']
        user = result_frame.loc[record_number]['username']
        message = result_frame.loc[record_number]['message']
        categoryframe = pd.DataFrame(json.loads(result_frame.loc[record_number]['category']), \
                                     columns = ['id', 'category', 'value'])
        subcategoryframe = pd.DataFrame(json.loads(result_frame.loc[record_number]['subcategory']), \
                                        columns = ['sbj', 'top_words', 'value'])
        executorframe = pd.DataFrame([json.loads(result_frame.loc[record_number]['executor'])])
        top_words = result_frame.loc[record_number]['topwords']
        datetime = result_frame.loc[record_number]['datetime']

        st.markdown('------------------')
        st.markdown('Номер записи: ' + str(index))
        st.markdown('Внес пользователь: ' + user)
        st.markdown('Дата сохранения: ' + str(datetime))
        st.markdown('### Текст обращения: ')
        st.write(message)
        
        st.markdown('### Топ-3 наиболее вероятных категорий:')
        st.write(categoryframe)
        chart = alt.Chart(categoryframe).mark_bar(size=30).encode(x='value', y='category', color='value').properties(width=700, height = 300)
        st.altair_chart(chart)
        
        st.markdown('### Топ-3 из 10 подкатегорий:')
        st.write(subcategoryframe)
        chart_2 = alt.Chart(subcategoryframe).mark_bar(size=30).encode(x='value', y='top_words', color='value').properties(width=700, height = 300)
        st.altair_chart(chart_2) 
        
        st.markdown('### Топ-3 наиболее вероятных исполнителей:')
        st.write(executorframe)
        ex = executorframe.T
        ex.columns = ['value']
        ex['name'] = ex.index
        ex.reset_index(drop=True, inplace=True)
        chart_3 = alt.Chart(ex).mark_bar(size=30).encode(x='value', y='name', color='value').properties(width=700, height = 300)
        st.altair_chart(chart_3)
        
        st.markdown('### Ключевые слова наиболее вероятной подкатегории:')
        st.write(top_words)

        st.write('------------------')

    return

form = st.sidebar.form(key='my_form')

#logins = load_username()
form.markdown("### 🎲 Авторизация")
#login = form.selectbox("Перед началом работы, представьтесь: ", logins)
login = form.text_input("Перед началом работы, представьтесь: ", "test")
password = form.text_input("Введите пароль: " , "test", type = "password")
new_button = form.form_submit_button(label='Создать новый аккаунт')

form_2 = st.sidebar.form(key='form_2')
num_count = form_2.selectbox("Вывести мои последние запросы в количестве: ", [1, 5, 10, 20, 50, 100])
display_button = form_2.form_submit_button(label='Отобразить')

form_3 = st.sidebar.form(key='form_3')
statistic_button = form_3.form_submit_button(label='Статистика')
delete_button = form_3.form_submit_button(label='Удалить запросы')

main_form = st.form(key='main_form')
message = main_form.text_input("Введите текст обращения:" , '')
analize_button = main_form.form_submit_button(label='Анализировать')

if new_button:
    keys = {'username':login, 'password': password}
    r = requests.post(r"http://" + serverhost + ":" + serverport + r"/newuser", params=keys)
        
    raw_json = json.loads(r.content.decode('utf-8'))
    
    if raw_json == []:
        st.write('Данные отсутствуют')
    elif raw_json['result'] != 'SUCCESS':
        st.write('Невозможно выполнить запрос!')
        st.write(raw_json)
    else:
        st.write('Пользователь успешно добавлен.')

if analize_button:
    if len(message) > 3:
        keys = {'username':login, 'password': password}
        body = json.dumps({'mess' : message}).encode('utf-8') #преобразование строки в байты
        r = requests.post(r"http://" + serverhost + ":" + serverport + r"/message", params=keys, data = body)
        
        raw_json = json.loads(r.content.decode('utf-8'))
        
        if raw_json == []:
            st.write('Данные отсутствуют')
        elif raw_json['result'] != 'SUCCESS':
            st.write('Невозможно выполнить запрос!')
            st.write(raw_json)
        else:
            write_content(raw_json)
    else:
        st.write('Ваш запрос некорректен и не был отправлен на сервер')

if display_button:
    keys = {'username':login, 'password': password, 'count': num_count}
    r = requests.get(r"http://" + serverhost + ":" + serverport + r"/message", params=keys)
    
    raw_json = json.loads(r.content.decode('utf-8'))
    
    if raw_json == []:
        st.write('Данные отсутствуют')
    elif raw_json['result']  == 'Некорректный логин или пароль!':
        st.write('Невозможно выполнить запрос!')
        st.write(raw_json)
    else:
        keys = ['index' , 'username', 'message' , 'category','subcategory' , 'executor', 'topwords', 'datetime', 'isvisible']
        result_dict = {key : [] for key in keys}
    
        for row in raw_json['result']:
            for i, key in enumerate(keys):
                result_dict[key].append(row[i])
        result_frame = pd.DataFrame(result_dict)
        write_contents(result_frame)

if statistic_button:
    keys = {'username':login, 'password': password}

    r = requests.get(r"http://" + serverhost + ":" + serverport + r"/statistics", params=keys)
    raw_json = json.loads(r.content.decode('utf-8'))
    result = raw_json['result']
    if result == 'Некорректный логин или пароль!' or result == []:
        st.write('Невозможно выполнить запрос!')
        st.write(raw_json)
    else:
        df = pd.DataFrame([result], index = ['Число запросов'])
        st.markdown('### Статистика запросов пользователей:')
        st.write(df)
        
if delete_button:
    keys = {'username':login, 'password': password}

    r = requests.delete(r"http://" + serverhost + ":" + serverport + r"/message", params=keys)
    raw_json = json.loads(r.content.decode('utf-8'))
    result = raw_json['result']
    if result == 'Некорректный логин или пароль!' or result == []:
        st.write('Невозможно выполнить запрос!')
        st.write(raw_json)
    elif result =='SUCCESS':
        if login == 'admin':
            st.markdown('### Запросы всех пользователей удалены')
        else:
            st.markdown('### Все ваши предыдущие запросы скрыты, но останутся видны для администратора')