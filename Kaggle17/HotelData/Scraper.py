# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:34:51 2017

@author: Jiashen Liu

@Purpose: Booking 
"""

import requests
from bs4 import BeautifulSoup
#import json
import re
import pandas as pd
import time
from datetime import timedelta

current_date=time.strftime("%Y-%m-%d")
Hotels = pd.read_csv('HotelList.csv',encoding='Latin-1')
url_list = list(Hotels['URL'])
Name_list = list(Hotels['Name'])
benchmark = timedelta(days = 730) ## Two years of data

def get_review_page(url):
    request = requests.get(url)
    content = request.content
    soup  = BeautifulSoup(content,'lxml')
    review_url = soup.select('.show_all_reviews_btn')[0].get('href')
    review_url = 'https://www.booking.com'+review_url
    num = int(soup.select('.show_all_reviews_btn')[0].getText().split(' ')[0])
    return review_url,num


def get_numbers(lst):
    final = []
    for each in lst:
        try:
            final.append(float(each))
        except Exception:
            pass
    return final          

def count_words(data):
    if data =='No Negative' or data == 'No Positive':
        return 0
    else:
        list_ = data.split(' ')
        return len(list_)


def get_clean_data(data):
    data = re.sub('\n',' ',data)
    data = re.sub('[^A-Za-z0-9]+', ' ', data)
    data = data.split(' ')
    data = [each for each in data if each!=' ']
    data =' '.join(data)
    return data


def get_score(data):
    data = re.sub('\n',' ',data)
    return float(data)

def get_review_num(data):
    data = data.split(' ')[1]
    data = int(data)
    return data

def get_reviews(url):
    url_level2,num_page = get_review_page(url)
    num_page = round(num_page/75)
    Name = []
    Natio = []
    ReviewNum = []
    Score = []
    Summary = []
    Tag_list = []
    Negative = []
    Positive = []
    Date = []
    Hotels = []
    Addresses = []
    Total_Reviews = []
    General_Scores = []
    Count_Neg = []
    Count_Pos = []
    ## For getting basic information
    url_1 = url_level2+'?page=1&'
    soup = BeautifulSoup(requests.get(url_1).content,'lxml')
    lst = soup.find('meta',{'name':'description'})['content'].split(' ')
    Number =get_numbers(lst)
    Total_Scores = Number[0]
    General_Score = Number[-2]
    Address = get_clean_data(soup.select('.hotel_address')[0].getText())
    Hotel = get_clean_data(soup.find('meta',{'itemprop':'name'})['content'])
    for i in range(1,num_page+1):
        print('Scraping page '+str(i))
        url_2 = url_level2+'?page='+str(i)+'&'
        requests_l2 = requests.get(url_2)
        content_l2 = requests_l2.content
        soup_l2 = BeautifulSoup(content_l2,'lxml')
        Header = soup_l2.select('.review_item_header_content')
        Reviewer = soup_l2.select('.review_item_reviewer')
        Content = soup_l2.select('.review_item_review_content')
        Bullet = soup_l2.select('.review_item_info_tags')
        Time = soup_l2.select('.review_item_date')
        Review_score = soup_l2.select('.review_item_review')
        no_text = soup_l2.select('.reviews_without_text')
        time_limit = soup_l2.select('.archived_separator')
        for j in range(len(Header)):
            Hotels.append(Hotel)
            Addresses.append(Address)
            Total_Reviews.append(Total_Scores)
            General_Scores.append(float(General_Score))
            Summary.append(get_clean_data(Header[j].getText()))
            Name.append(get_clean_data(Reviewer[j].find('span').getText()))
            Natio.append(get_clean_data(Reviewer[j].select('.reviewer_country')[0].getText()))
            try:
                ReviewUser=Reviewer[j].select('.review_item_user_review_count')[0].getText()
            except Exception:
                ReviewUser=' 0 Reviews'
            ReviewNum.append(get_review_num(get_clean_data(ReviewUser)))
            Neg = Content[j].select('.review_neg')
            Neg = Neg[0].getText() if len(Neg)>0 else 'No Negative'
            Pos = Content[j].select('.review_pos')
            Pos = Pos[0].getText() if len(Pos)>0 else 'No Positive'
            Neg = get_clean_data(Neg)
            Pos = get_clean_data(Pos)
            Negative.append(Neg)
            Positive.append(Pos)
            Count_Neg.append(count_words(Neg))
            Count_Pos.append(count_words(Pos))
            tags = Bullet[j].select('.review_info_tag')
            tags = [get_clean_data(each.getText()) for each in tags]
            Tag_list.append(tags)
            Date.append(get_clean_data(Time[j].getText()))
            Score.append(get_score(Review_score[j].select('meta')[0]['content']))
        if len(no_text)>0 or len(time_limit)>0:
            break
    try:
        additional_number = int(no_text[0].getText().split(' ')[3])
    except Exception:
        additional_number = 0
    Additional_review = [additional_number]*len(Hotels)
    cdate = [current_date]*len(Hotels)
    df = pd.DataFrame({'Scraping_Date':cdate,'Hotel':Hotels,'Address':Addresses,'NumberReviews':Total_Reviews,'General_Score':General_Scores,'Name':Name,'Nationality':Natio,'Date':Date,'Score':Score,'Negative':Negative,'Positive':Positive,'Tags':Tag_list,'Review_Num':ReviewNum,'Negative_Counts':Count_Neg,'Positive_Counts':Count_Pos,'Additional_Number':Additional_review})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Scraping_Date'] = pd.to_datetime(df['Scraping_Date'])
    df['date_gap']=df['Scraping_Date'] - df['Date']
    df = df[df['date_gap']<=benchmark]
    return df

for i in range(5):#range(len(url_list)): #
    Abondoned_Name = []
    IDs = []
    url = url_list[i]
    Hotel_name = Name_list[i]
    try:
        print('Start Scraping '+Hotel_name)
        Reviews = get_reviews(url)
    except Exception:
        Abondoned_Name.append(Hotel_name)
        print('We need to abondon '+Hotel_name)
        IDs.append(i)
    if i==0:
        Final_Review = Reviews
    else:
        Final_Review = pd.concat([Final_Review,Reviews])
    if i%10==0:
        Final_Review.to_csv('tmp.csv',index=False)
Final_Review.to_csv('Hotel_Reviews.csv',index=False) 
Abondon = pd.DataFrame({'ID':IDs,'Name':Abondoned_Name})
Abondon.to_csv('Abondoned_list.csv',index=False)   
    
