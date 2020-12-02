# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:54:29 2020
@author: YuL
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

'''
Load Netflix data (Final output df & df_title)
'''
nf_path='data/netflix'
os.chdir(nf_path)

df1 = pd.read_csv('combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating','date'])
df1['Rating'] = df1['Rating'].astype(float)
df1['date']=pd.to_datetime(df1['date'])

df2 = pd.read_csv('combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating','date'])
df2['Rating'] = df2['Rating'].astype(float)
df2['date']=pd.to_datetime(df2['date'])

df3 = pd.read_csv('combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating','date'])
df3['Rating'] = df3['Rating'].astype(float)
df3['date']=pd.to_datetime(df3['date'])

df4 = pd.read_csv('combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating','date'])
df4['Rating'] = df4['Rating'].astype(float)
df4['date']=pd.to_datetime(df4['date'])

#Combine df1-4
df = df1
df = df1.append(df2)
df = df.append(df3)
df = df.append(df4)
df.index = np.arange(0,len(df))

#Remove df with no rating
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

#correct the dataframe of db
movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

#Add movie id
df = df[pd.notnull(df['Rating'])]
df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)

#Add Movie title table, add another colume without special charactors
df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title['Name1']=df_title['Name'].str.replace(" ","")
df_title['Name1']=df_title.Name1.str.replace('\W', '')
df_title['Name1']=df_title['Name1'].str.lower()
df_title['Year']=df_title['Year'].astype(str)
df_title['Year']=df_title['Year'].str[:4]

'''
Load IMDB Data (Final output t3)
'''
#Load imdb table
imdb_path='data/imdb'
os.chdir(imdb_path)

t1=pd.read_csv('name_basics.tsv',sep='\t')
t2=pd.read_csv('title_akas.tsv',sep='\t')
t3=pd.read_csv('title_basics.tsv',sep='\t')
t4=pd.read_csv('title_crew.tsv',sep='\t')
t5=pd.read_csv('title_episode.tsv',sep='\t')
t6=pd.read_csv('title_principals.tsv',sep='\t')
t7=pd.read_csv('title_ratings.tsv',sep='\t')

t3['Name1']=t3['primaryTitle'].str.replace(" ","")
t3['Name1']=t3.Name1.str.replace('\W', '')
t3['Name1']=t3['Name1'].str.lower()

'''
Merge IMDB with Netflix movie title (final output df_title1 & movie for movie_id)
'''
#merge data
df_title1=df_title.merge(t3, left_on=['Name1','Year'], right_on=['Name1','startYear'] ,how='inner')
df_title1=df_title1.merge(t7, left_on=['tconst'], right_on=['tconst'] ,how='left')

#Seperate movie type
s=set()
for i in range(df_title1.shape[0]):
    l=df_title1['genres'][i].split(",")
    for j in range(len(l)):
        s.add(l[j])
        j+=1
    i+=1

df_title1=df_title1.reindex(columns=df_title1.columns.tolist()+list(s))

for i in range(len(s)):
    df_title1[list(s)[i]]=df_title1['genres'].map(lambda x: 1 if list(s)[i] in x else 0)
    i+=1

Movie=df_title1.drop(columns=['primaryTitle','endYear','originalTitle','isAdult','startYear'])

unique_movie=Movie['Movie_Id']

'''
Limit data by customer history (due to computation cost) (output Netflix final output: customer)
'''
#Find the customer information (limit data)
cust1=df.groupby(['Cust_Id']).agg({'date':[np.max,np.min]}).reset_index()
cust1.columns=['Cust_Id','date_max','date_min']
cust1['mon_max']=cust1['date_max'].str[:7]
cust1['mon_max']=cust1['mon_max'].str[-2:].astype(int)
cust1['mon_min']=cust1['date_min'].str[:7]
cust1['mon_min']=cust1['mon_min'].str[-2:].astype(int)
cust1['year_min']=cust1['date_min'].str[:4].astype(int)
cust1['year_max']=cust1['date_max'].str[:4].astype(int)
cust1['year_max'].value_counts().plot(kind='bar')
plt.show()
cust1.head()

cust_new=cust1[(cust1['year_min']==2003) & (cust1['mon_min']==1) & (cust1['mon_max']>=11) & (cust1['year_max']==2005)]
cust_new1=cust_new['Cust_Id']

df_t=df.merge(cust_new1,left_on=['Cust_Id'],right_on=['Cust_Id'], how='inner')

customer=df_t.merge(unique_movie,left_on=['Movie_Id'],right_on=['Movie_Id'],how='inner')


'''
Load oscar Data  (Final output oscar1)
'''
oscar=pd.read_csv('data/oscar.csv')

oscar['film1']=oscar['Film'].str.replace(" ","")
oscar['film1']=oscar.film1.str.replace('\W', '')
oscar['film1']=oscar['film1'].str.lower()

oscar=oscar.merge(Movie['Name1'],left_on='film1',right_on='Name1',how='inner')
oscar['oscar']=1

oscar1=oscar.groupby(['Name1','oscar']).agg({'Winner':[np.max]}).reset_index()

oscar1=oscar1.fillna(0)
oscar1.columns=['Name1','oscar','winner']

'''
Build final table
'''
total=customer.merge(Movie, left_on='Movie_Id',right_on='Movie_Id',how='inner')

total1=total.merge(oscar1, left_on='Name1',right_on='Name1',how='left')

'''
Predict next movie
'''

'''
Learning the traning data
'''
###Train split for Predict next movie
total1['year']=total1['date'].str[:4].astype(int)
total1['month']=total1['date'].str[:7]
total1['month']=total1['month'].str[-2:].astype(int)

total_new_train=total1[((total1['year']==2005) & (total1['month']<11)) | (total1['year']<=2005)]
total_new_test=total1[(total1['year']==2005) & (total1['month']>10)]

train=total_new_train[['Cust_Id']]
train_movie=total_new_train[['Cust_Id','Movie_Id']]
train_movie['Movie_Id']=train_movie['Movie_Id'].astype(str)
train_movie['Watch']=1

y_train=total_new_test.groupby('Cust_Id').first().reset_index()
y_train1=y_train[['Cust_Id','Movie_Id']]
y_train1['Movie_Id']=y_train1['Movie_Id'].astype(str)

#Cal weight for each type of data
def cal_weight1(db,l):
    for i in range(len(l)):
       db[l[i]]=db[l[i]]/db['movie_count']
       i+=1
    l=['Cust_Id']+l+['movie_count']
    db1=db[l]
    return db1

#movie type watch frequency
train1_movie_type=total_new_train.groupby(['Cust_Id']).agg({'Crime':[np.sum], 'Action':[np.sum], 'Animation':[np.sum],
       'Horror':[np.sum], 'Fantasy':[np.sum], 'Musical':[np.sum], 'Adult':[np.sum], 'Comedy':[np.sum], 'Thriller':[np.sum],
       'Talk-Show':[np.sum], 'War':[np.sum], 'Music':[np.sum], 'News':[np.sum], 'Drama':[np.sum], 'Sci-Fi':[np.sum], 'Sport':[np.sum],
       'Western':[np.sum], 'Game-Show':[np.sum], 'Short':[np.sum], 'Reality-TV':[np.sum], 'Adventure':[np.sum], 'Biography':[np.sum],
       'Romance':[np.sum], 'History':[np.sum], 'Documentary':[np.sum], 'Family':[np.sum], 'Mystery':[np.sum],'Movie_Id':['count']}).reset_index()

train1_movie_type.columns=['Cust_Id','Crime', 'Action', 'Animation',
       'Horror', 'Fantasy', 'Musical', 'Adult', 'Comedy', 'Thriller',
       'Talk-Show', 'War', 'Music', 'News', 'Drama', 'Sci-Fi', 'Sport',
       'Western', 'Game-Show', 'Short', 'Reality-TV', 'Adventure', 'Biography',
       'Romance', 'History', 'Documentary', 'Family', 'Mystery','movie_count']

movie_type=['Crime', 'Action', 'Animation',
       'Horror', 'Fantasy', 'Musical', 'Adult', 'Comedy', 'Thriller',
       'Talk-Show', 'War', 'Music', 'News', 'Drama', 'Sci-Fi', 'Sport',
       'Western', 'Game-Show', 'Short', 'Reality-TV', 'Adventure', 'Biography',
       'Romance', 'History', 'Documentary', 'Family', 'Mystery']

train2_movie_type=cal_weight1(train1_movie_type,movie_type)

#oscar nominate or winning
train1_oscar=total_new_train.groupby(['Cust_Id']).agg({ 'oscar':[np.sum],'winner':[np.sum],'Movie_Id':['count']}).reset_index()
oscar=['oscar','winner']
train1_oscar.columns=['Cust_Id']+oscar+['movie_count']

train2_oscar=cal_weight1(train1_oscar,oscar)

#Movie overall score
train_score=total_new_train[['Cust_Id','Movie_Id','averageRating','numVotes']]

train_score["averageRating"].fillna(0 ,inplace = True) 

train_score['0-5']=train_score['averageRating'].map(lambda x: 1 if round(x)<=5 else 0)
train_score['6-7']=train_score['averageRating'].map(lambda x: 1 if (round(x)<=7) & (round(x)>=6) else 0)
train_score['8-10']=train_score['averageRating'].map(lambda x: 1 if round(x)>=8 else 0)

train1_score=train_score.groupby(['Cust_Id']).agg({'0-5':[np.sum],'6-7':[np.sum],'8-10':[np.sum],'Movie_Id':['count']}).reset_index()
train1_score.columns=['Cust_Id','0-5','6-7','8-10','movie_count']

score=['0-5','6-7','8-10']
train2_score=cal_weight1(train1_score,score)

#movie popularity
cust_movie=total_new_train[['Movie_Id','numVotes']].drop_duplicates()
q1=cust_movie['numVotes'].quantile(0.25)
q2=cust_movie['numVotes'].quantile(0.5)
q3=cust_movie['numVotes'].quantile(0.75)


print(q1,q2,q3)

train_score['popular_1']=train_score['numVotes'].map(lambda x: 1 if x<q1 else 0)
train_score['popular_2']=train_score['numVotes'].map(lambda x: 1 if (x>=q1) & (x<q2) else 0)
train_score['popular_3']=train_score['numVotes'].map(lambda x: 1 if (x>=q2) & (x<q3) else 0)
train_score['popular_4']=train_score['numVotes'].map(lambda x: 1 if x>=q3 else 0)

train1_popular=train_score.groupby(['Cust_Id']).agg({'popular_1':[np.sum],'popular_2':[np.sum],'popular_3':[np.sum],'popular_4':[np.sum],'Movie_Id':['count']}).reset_index()
train1_popular.columns=['Cust_Id','popular_1','popular_2','popular_3','popular_4','movie_count']

popular=['popular_1','popular_2','popular_3','popular_4']

train2_popular=cal_weight1(train1_popular,popular)

#movie year
train_year=total_new_train[['Cust_Id','Movie_Id','Year']]
train_year=train_year.sort_values(['Year']).reset_index(drop=True)

train_year['2005']=train_year['Year'].map(lambda x: 1 if x>=2005 else 0)
train_year['03-04']=train_year['Year'].map(lambda x: 1 if (x>=2003) & (x<2005) else 0)
train_year['00-02']=train_year['Year'].map(lambda x: 1 if (x>=2000) & (x<=2002) else 0)
train_year['90s']=train_year['Year'].map(lambda x: 1 if (x>=1990) & (x<=1999) else 0)
train_year['80s']=train_year['Year'].map(lambda x: 1 if (x>=1980) & (x<=1989) else 0)
train_year['70s']=train_year['Year'].map(lambda x: 1 if (x>=1970) & (x<=1979) else 0)
train_year['60s-']=train_year['Year'].map(lambda x: 1 if x<=1969 else 0)

year=['2005','03-04','00-02','90s','80s','70s','60s-']

train1_year=train_year.groupby(['Cust_Id']).agg({'2005':[np.sum],'03-04':[np.sum],'00-02':[np.sum],'90s':[np.sum],'80s':[np.sum],'70s':[np.sum],'60s-':[np.sum],'Movie_Id':['count']}).reset_index()
train1_year.columns=['Cust_Id']+year+['movie_count']

train2_year=cal_weight1(train1_year,year)

'''
Table for all the available movie
'''
Movie1=Movie.merge(oscar1,left_on='Name1',right_on='Name1',how='left')
Movie1=Movie1.fillna(0)

#Traning output-type
mov1_type=Movie1[['Movie_Id','genres']]
mov1_type=mov1_type.reindex(columns=mov1_type.columns.tolist()+list(movie_type))
for i in range(len(movie_type)):
    mov1_type[movie_type[i]]=mov1_type['genres'].map(lambda x: 1 if movie_type[i] in x else 0)
    i+=1

#Traning output-oscar
mov1_oscar=Movie1[['Movie_Id','oscar','winner']]

#Training output-avg score
mov1_score=Movie1[['Movie_Id','averageRating']]
mov1_score=mov1_score.reindex(columns=mov1_score.columns.tolist()+list(score))

mov1_score['0-5']=mov1_score['averageRating'].map(lambda x: 1 if round(x)<=5 else 0)
mov1_score['6-7']=mov1_score['averageRating'].map(lambda x: 1 if (round(x)<=7) & (round(x)>=6) else 0)
mov1_score['8-10']=mov1_score['averageRating'].map(lambda x: 1 if round(x)>=8 else 0)

#Traning output popularity
mov1_popular=Movie1[['Movie_Id','numVotes']]
mov1_popular=mov1_popular.reindex(columns=mov1_popular.columns.tolist()+list(popular))
mov1_popular['popular_1']=mov1_popular['numVotes'].map(lambda x: 1 if x<q1 else 0)
mov1_popular['popular_2']=mov1_popular['numVotes'].map(lambda x: 1 if (x>=q1) & (x<q2) else 0)
mov1_popular['popular_3']=mov1_popular['numVotes'].map(lambda x: 1 if (x>=q2) & (x<q3) else 0)
mov1_popular['popular_4']=mov1_popular['numVotes'].map(lambda x: 1 if x>=q3 else 0)

#Training output year
mov1_year=Movie1[['Movie_Id','Year']]
mov1_year=mov1_year.reindex(columns=mov1_year.columns.tolist()+list(year))
mov1_year['2005']=mov1_year['Year'].map(lambda x: 1 if x>=2005 else 0)
mov1_year['03-04']=mov1_year['Year'].map(lambda x: 1 if (x>=2003) & (x<2005) else 0)
mov1_year['00-02']=mov1_year['Year'].map(lambda x: 1 if (x>=2000) & (x<=2002) else 0)
mov1_year['90s']=mov1_year['Year'].map(lambda x: 1 if (x>=1990) & (x<=1999) else 0)
mov1_year['80s']=mov1_year['Year'].map(lambda x: 1 if (x>=1980) & (x<=1989) else 0)
mov1_year['70s']=mov1_year['Year'].map(lambda x: 1 if (x>=1970) & (x<=1979) else 0)
mov1_year['60s-']=mov1_year['Year'].map(lambda x: 1 if x<=1969 else 0)

'''
Score movie by customer preference
'''
final_score=cust_new[['Cust_Id']].drop_duplicates(subset='Cust_Id', keep='first', inplace=False)
final_score=final_score.reindex(columns=final_score.columns.tolist()+list(Movie['Movie_Id']))

def type_score(db):
    for i in range(db.shape[0]):
        for j in range(Movie.shape[0]):
            print(i,j)
            t=train2_movie_type[movie_type].iloc[i].dot(mov1_type[movie_type].iloc[j])
            db[Movie['Movie_Id'].iloc[j]].iloc[i]=t
            j+=1
        i+=1
    return db

def oscar_score(db):
    for i in range(db.shape[0]):
        for j in range(Movie.shape[0]):
            print(i,j)
            o=train2_oscar[oscar].iloc[i].dot(mov1_oscar[oscar].iloc[j])
            db[Movie['Movie_Id'].iloc[j]].iloc[i]=o
            j+=1
        i+=1
    return db

def score_score(db):
    for i in range(db.shape[0]):
        for j in range(Movie.shape[0]):
            print(i,j)
            s=train2_score[score].iloc[i].dot(mov1_score[score].iloc[j])
            db[Movie['Movie_Id'].iloc[j]].iloc[i]=s
            j+=1
        i+=1
    return db

def popular_score(db):
    for i in range(db.shape[0]):
        for j in range(Movie.shape[0]):
            print(i,j)
            p=train2_popular[popular].iloc[i].dot(mov1_popular[popular].iloc[j])
            db[Movie['Movie_Id'].iloc[j]].iloc[i]=p
            j+=1
        i+=1
    return db

def year_score(db):
    for i in range(db.shape[0]):
        for j in range(Movie.shape[0]):
            print(i,j)
            y=train2_year[year].iloc[i].dot(mov1_year[year].iloc[j])
            db[Movie['Movie_Id'].iloc[j]].iloc[i]=y
            j+=1
        i+=1
    return db

final_score_type=type_score(final_score)
final_score_type1=final_score_type.melt(id_vars=['Cust_Id'],var_name='Movie_Id',value_name='type')

final_score_oscar=oscar_score(final_score)
final_score_oscar1=final_score_oscar.melt(id_vars=['Cust_Id'],var_name='Movie_Id',value_name='oscar')

final_score_score=score_score(final_score)
final_score_score1=final_score_score.melt(id_vars=['Cust_Id'],var_name='Movie_Id',value_name='score')

final_score_popular=popular_score(final_score)
final_score_popular1=final_score_popular.melt(id_vars=['Cust_Id'],var_name='Movie_Id',value_name='popular')

final_score_year=year_score(final_score)
final_score_year1=final_score_year.melt(id_vars=['Cust_Id'],var_name='Movie_Id',value_name='year')

final_table=final_score_type1.merge(final_score_oscar1,left_on=['Cust_Id','Movie_Id'],right_on=['Cust_Id','Movie_Id'],how='inner')
final_table=final_table.merge(final_score_score1,left_on=['Cust_Id','Movie_Id'],right_on=['Cust_Id','Movie_Id'],how='inner')
final_table=final_table.merge(final_score_popular1,left_on=['Cust_Id','Movie_Id'],right_on=['Cust_Id','Movie_Id'],how='inner')
final_table=final_table.merge(final_score_year1,left_on=['Cust_Id','Movie_Id'],right_on=['Cust_Id','Movie_Id'],how='inner')

final_table=final_table.merge(train_movie,left_on=['Cust_Id','Movie_Id'],right_on=['Cust_Id','Movie_Id'],how='left')
final_table['total_score']=final_table['type']+final_table['oscar']+final_table['score']+final_table['popular']+final_table['year']
final_table=final_table.fillna(0)

##final_table.to_csv('final_table_v1.csv')
'''
Select the top n highest unwatched movie for each customer
'''
#define function to calculate accuracy
def Top_n_recommend(df,field,n):
    n=int(n)
    df=df[df['Watch']!=1]
    t=train[['Cust_Id']]
    d={}
    for i in range(t.shape[0]):
        df3=df[df['Cust_Id']==t['Cust_Id'].iloc[i]]
        df3=df3.nlargest(n,[field])
        d[t['Cust_Id'].iloc[i]]=df3['Movie_Id'].values.tolist()
        i+=1
    return d

def cal_accuracy(df,field,n,df2):
    d=Top_n_recommend(df,field,n)
    r=0
    t=0
    for key in d.keys():
        a=df2.loc[df2['Cust_Id'] == key]
        a1=a.iloc[0,1]
        if a1 in d[key]:
            r+=1
            t+=1
        else:
            r=r
            t+=1
    return r,t

for i in range(1,11):
    acc=cal_accuracy(final_table,'total_score',10*i,y_train1)
    print(acc)

#calculate weight for each inputs via logistic regression model
lr=LogisticRegression()
def lr_weight(df):
    t=train[['Cust_Id']]
    t=t.reindex(columns=t.columns.tolist()+list(['type','oscar','score','popular','year']))
    for i in range(t.shape[0]):
        df1=df[df['Cust_Id']==t['Cust_Id'].iloc[i]]
        lr_x=df1[['type','oscar','score','popular','year']]
        lr_y=df1[['Watch']]
        lr1=lr.fit(lr_x,lr_y)
        a=lr1.coef_.tolist()
        l=a[0]
        for j in range(5):
            t.iloc[i,j+1]=l[j]
            j+=1
        i+=1
    return t

lr_w=lr_weight(final_table)
final_table_lr=final_table.reindex(columns=final_table.columns.tolist()+['lr_total'])
lr_w=lr_w[['Cust_Id','type','oscar','score','popular','year']]
lr_w.columns=['Cust_Id','type1','oscar1','score1','popular1','year1']

final_table_lr=final_table.merge(lr_w, left_on=['Cust_Id'],right_on=['Cust_Id'],how='left')
final_table_lr['lr_total']=final_table_lr['type']*final_table_lr['type1']+final_table_lr['oscar']*final_table_lr['oscar1']+final_table_lr['score']*final_table_lr['score1']+final_table_lr['popular']*final_table_lr['popular1']+final_table_lr['year']*final_table_lr['year1']
final_table_lr

for i in range(1,11):
    lr_acc=cal_accuracy(final_table_lr,'lr_total',10*i,y_train1)
    print(lr_acc)

def data_for_ae(df,df1,l):
    df_1=df
    df_1['t']=0
    df1_1=df1
    df1_1['t']=1
    df2=df_1.append(df1_1)
    scaler=MinMaxScaler()
    df2[l]=scaler.fit_transform(df2[l])
    df2=df2.drop(columns=['Cust_Id','Movie_Id','date','Rating','t'])
    return df2

def ae(df,n):
    input_dim = df.shape[1]
    encoding_dim=n
    input_layer = Input(shape=(input_dim, ))
    encoder_layer_1 = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder=Model(inputs=input_layer,outputs=encoder_layer_1)
    ec_train=pd.DataFrame(encoder.predict(df))
    return ec_train

def data_out_ae(df,df1):
    df_1=df
    df_1['t']=0
    df1_1=df1
    df1_1['t']=1
    df3=df_1.append(df1_1)
    df3=df3[['Cust_Id','Movie_Id','date','Rating','t']]
    #df3=pd.concat([df3,df2],axis=0)
    return df3

#dataset 1
train1=total_new_train[['Cust_Id','date','Movie_Id','Year','averageRating','numVotes','Animation','Romance','Sci-Fi','Family','Biography','Fantasy','Documentary','Western','Mystery','Action','Music','Comedy','News','Game-Show','Adult','Thriller','War','Reality-TV','Talk-Show','Drama','Sport','Adventure','History','Crime','Short','Musical','Horror','oscar','winner','Rating']]
train1=train1.fillna(0)
train1[‘t’]=0
train1['Rating']=train1['Rating'].astype(int)

test1=total_new_test[['Cust_Id','date','Movie_Id','Year','averageRating','numVotes','Animation','Romance','Sci-Fi','Family','Biography','Fantasy','Documentary','Western','Mystery','Action','Music','Comedy','News','Game-Show','Adult','Thriller','War','Reality-TV','Talk-Show','Drama','Sport','Adventure','History','Crime','Short','Musical','Horror','oscar','winner','Rating']]
test1=test1.fillna(0)
test[‘t’]=1
test1['Rating']=test1['Rating'].astype(int)

#dataset 2
l=['Year','averageRating','numVotes','Animation','Romance','Sci-Fi','Family','Biography','Fantasy','Documentary','Western','Mystery','Action','Music','Comedy','News','Game-Show','Adult','Thriller','War','Reality-TV','Talk-Show','Drama','Sport','Adventure','History','Crime','Short','Musical','Horror','oscar','winner']
df1=data_for_ae(train1,test1,l)
ec=ae(df1,2)
df1=data_out_ae(train1,test1)
ec.columns=['f1','f2']
df1=pd.concat([df1.reset_index(),ec.reset_index()],axis=1)
train2=df1[df1[‘t’]==0].drop(columns=[‘t’])
test2=df1[df1[‘t’]==1].drop(columns=[‘t’])

#dataset 3
ft=final_table[final_talbe[‘Watch’]==1]
train3=train1[['Cust_Id','Movie_Id',’date’,'Rating']]
test3=test1[['Cust_Id','Movie_Id',’date’,'Rating']]
train3=ft.merge(train3,on=['Cust_Id','Movie_Id'],how='inner')
test3=ft.merge(test3,on=['Cust_Id','Movie_Id'],how='inner')

#dataset 4
l1=[‘type’,’year’,’oscar’,’popular’,’score’]
df2=data_for_ae(train3,test3,l1)
ec1=ae(df2,1)
df2=data_out_ae(train3,test3)
ec1.columns=['f1']
df2=pd.concat([df2.reset_index(),ec1.reset_index()],axis=1)
train4=df2[df2[‘t’]==0].drop(columns=[‘t’])
test4=df2[df2[‘t’]==1].drop(columns=[‘t’])

#Caulcate model accuracy
def model_acc(m,train,test):
  acc=0
  pred=0
  for i in range(cust.shape[0]):
    x=train[train['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,3:-1]
    x_t=test[test['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,3:-1]
    y=train[train['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,-1]
    y_t=test[test['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,-1]
    m.fit(x,y)
    y_pred=m.predict(x_t)
    for j in range(len(y_pred)):
      pred+=1
      if y_pred[j]==y_t.iloc[j]:
        acc+=1
      else:
        acc=acc
      j+=1
    i+=1
  return acc/pred*100

#model
lr=LogisticRegression()
rf=RandomForestClassifier(random_state=1)
mlp=MLPClassifier(hidden_layer_sizes=(50,),random_state=1)
nb=GaussianNB()
knn=KNeighborsClassifier()

def lstm_acc(train,test):
    acc=0
    pred=0
    for i in range(cust.shape[0]):
        x=train[train['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,3:-1]
        x_t=test[test['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,3:-1]
        y=train[train['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,-1]
        y_t=test[test['Cust_Id']==cust['Cust_Id'].iloc[i]].iloc[:,-1]
        x=x.values.reshape((x.shape[0],1,x.shape[1]))
        x_t=x_t.values.reshape((x_t.shape[0],1,x_t.shape[1]))
        model=Sequential()
        model.add(LSTM(50,input_shape=(x.shape[1],x.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer='adam')
        model.fit(x,y,epochs=10,batch_size=10)
        y_pred=model.predict(x_t)
        for j in range(len(y_pred)):
            pred+=1
            if y_pred[j]==y_t.iloc[j]:
                acc+=1
            else:
                acc=acc
            j+=1
        i+=1
    return acc/pred*100


#dataset 1 - train
lr1=model_acc(lr,train1,train1)
rf1=model_acc(rf,train1,train1)
mlp1=model_acc(mlp,train1,train1)
np1=model_acc(nb,train1,train1)
knn1=model_acc(knn,train1,train1)

#dataset 1 - test
lr11=model_acc(lr,train1,test1)
rf11=model_acc(rf,train1,test1)
mlp11=model_acc(mlp,train1,test1)
np11=model_acc(nb,train1,test1)
knn11=model_acc(knn,train1,test1)

#dataset 2 - train
lr2=model_acc(lr,train2,train2)
rf2=model_acc(rf,train2,train2)
mlp2=model_acc(mlp,train2,train2)
np2=model_acc(nb,train2,train2)
knn2=model_acc(knn,train2,train2)

#dataset 2 - test
lr21=model_acc(lr,train2,test2)
rf21=model_acc(rf,train2,test2)
mlp21=model_acc(mlp,train2,test2)
np21=model_acc(nb,train2,test2)
knn21=model_acc(knn,train2,test2)

#dataset 3 - train
lr3=model_acc(lr,train3,train3)
rf3=model_acc(rf,train3,train3)
mlp3=model_acc(mlp,train3,train3)
np3=model_acc(nb,train3,train3)
knn3=model_acc(knn,train3,train3)

#dataset 3 - test
lr31=model_acc(lr,train3,test3)
rf31=model_acc(rf,train3,test3)
mlp31=model_acc(mlp,train3,test3)
np31=model_acc(nb,train3,test3)
knn31=model_acc(knn,train3,test3)

#dataset 4 - train
lr4=model_acc(lr,train4,train4)
rf4=model_acc(rf,train4,train4)
mlp4=model_acc(mlp,train4,train4)
np4=model_acc(nb,train4,train4)
knn4=model_acc(knn,train4,train4)

#dataset 4 - test
lr41=model_acc(lr,train4,test4)
rf41=model_acc(rf,train4,test4)
mlp41=model_acc(mlp,train4,test4)
np41=model_acc(nb,train4,test4)
knn41=model_acc(knn,train4,test4)