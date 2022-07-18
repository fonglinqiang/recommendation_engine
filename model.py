import time, ast, random
from datetime import date
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import strip_punctuation,strip_numeric,strip_multiple_whitespaces,remove_stopwords,split_on_space,strip_short
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from surprise import Reader, Dataset, SVD
from collections import defaultdict,Counter


# Preprocessing

def preprocessing(watch_list,df_media):
    watch_list_meta = pd.merge(watch_list,df_media[['id','title','type','duration','series_meta_series_id']].rename(columns={'id':'media_id'}),how='left',on='media_id')
    watch_list_meta_movie = watch_list_meta[watch_list_meta.type == 'movie']
    watch_list_meta_series = watch_list_meta[watch_list_meta.type != 'movie']

    series_id_mapping = df_media[df_media.type == 'series'][['title','series_meta_series_id']].set_index('series_meta_series_id')['title'].to_dict()
    series_uuid_mapping = df_media[df_media.type == 'series'][['id','series_meta_series_id']].set_index('series_meta_series_id')['id'].to_dict()

    watch_list_meta_series.sort_values(by=['user_id','series_meta_series_id'],inplace=True)

    watch_list_meta_series_processed = pd.DataFrame(columns=watch_list.columns)

    for user_id in watch_list_meta_series.user_id.unique():
        # print(user_id)
        watch_list_meta_series_user = watch_list_meta_series[watch_list_meta_series.user_id == user_id]
        for series_id in watch_list_meta_series_user.series_meta_series_id.unique():
            # print(series_id)
            watch_list_meta_series_episode = watch_list_meta_series_user[watch_list_meta_series_user.series_meta_series_id == series_id]
            if len(watch_list_meta_series_episode) == 0:
                continue
            watch_list_meta_series_episode_latest = watch_list_meta_series_episode.tail(1).index[0]
            watch_list_meta_series_series = {
                'id':watch_list_meta_series_episode.loc[watch_list_meta_series_episode_latest,'id'],
                'user_id':user_id, 
                'media_id':series_uuid_mapping[series_id], 
                'last_play_time_in_seconds':round(watch_list_meta_series_episode.last_play_time_in_seconds.mean()), 
                'completed':False,
                'continue_playing_active':True, 
                'history_record_active':True, 
                'created_date':watch_list_meta_series_episode.loc[watch_list_meta_series_episode_latest,'created_date'],
                'created_ts':watch_list_meta_series_episode.loc[watch_list_meta_series_episode_latest,'created_ts'], 
                'updated_date':None, 
                'updated_ts':None,
                'title':series_id_mapping[series_id],
                'type':'series',
                'duration': watch_list_meta_series_episode.loc[watch_list_meta_series_episode_latest,'duration'],
                'series_meta_series_id': series_id
                }
            watch_list_meta_series_processed = watch_list_meta_series_processed.append(watch_list_meta_series_series,ignore_index=True)

    watch_list_processed = pd.concat([watch_list_meta_movie,watch_list_meta_series_processed])
    print('Calculating ratings using play history')
    watch_list_processed['rating'] = watch_list_processed.apply(infer_ratings,axis=1)
    watch_list_rating = watch_list_processed[['user_id','media_id','rating']]
    return watch_list_rating


# Content-based

wn_dict = {'J':wordnet.ADJ,'V':wordnet.VERB,'N':wordnet.NOUN,'R':wordnet.ADV}

lemmatizer = WordNetLemmatizer()


def clean_text(text): # for description
    try:
        text = strip_punctuation(text.lower())
        text = strip_numeric(text)
        text = strip_multiple_whitespaces(text)
        text = remove_stopwords(text)
        text = strip_short(text)
        text = split_on_space(text)
        text = pos_tag(text)
        text = [lemmatizer.lemmatize(t,pos=wn_dict[tag[0]]) if tag[0] in ['J','N','R','V'] else lemmatizer.lemmatize(t) for t,tag in text ]
        return ' '.join(text)
    except:
        return ''


def clean_list(text): # for 'tags','genres','directors','cast'
    if type(text) == str:
        text = ast.literal_eval(text)
    if len(text) > 10:
        text = text[:10]
    text = [strip_punctuation(t).lower().replace(' ','',99) for t in text]
    return ' '.join(text)


def content_based_filtering(df):
    print('Running content-based filtering')
    df = df[df.type != 'episode'].copy()
    df_text = df[['id','title','description','tags','genres','directors','cast']]

    # Clean Text Data
    df_text['tags'] = df_text.tags.apply(clean_list)
    df_text['genres'] = df_text.genres.apply(clean_list)
    df_text['directors'] = df_text.directors.apply(clean_list)
    df_text['cast'] = df_text.cast.apply(clean_list)
    df_text['clean_text'] = df_text.description.apply(clean_text)
    df_text['final'] = df_text['clean_text'] + ' ' + df_text['tags'] + ' ' + df_text['genres'] + ' ' + df_text['directors'] + ' ' + df_text['cast']

    # TFIDF Vectorisation
    v = TfidfVectorizer()
    x = v.fit_transform(df_text['final'].tolist())
    x = torch.from_numpy(x.A)

    # Similarity Calculation
    sim_mat = np.zeros([len(df_text), len(df_text)])
    cos = nn.CosineSimilarity(dim=0,eps=1e-6)

    time_start = time.time()
    for i in range(len(df_text)):
        if i%1000 == 0 and i != 0:
            print(i, f'{(time.time() - time_start)/60} mins')
            time_start = time.time()
        for j in range(i):
            score_temp = float(cos(x[i],x[j]))
            sim_mat[i,j] = score_temp
            sim_mat[j,i] = score_temp
    df_score_byid = pd.DataFrame(sim_mat,index=df_text.id,columns=df_text.id)
    df_score_bytitle = pd.DataFrame(sim_mat,index=df_text.title,columns=df_text.title)

    # # Update Cached Similarity Score
    print('Caching similarity score')
    df_score_byid.to_csv('cached_data/cached_similarity_score_byid.csv')
    df_score_bytitle.to_csv('cached_data/cached_similarity_score_bytitle.csv')

    return df_score_byid,df_score_bytitle


def update_content_recommendation(watch_list_rating,read_user_list,df_score_byid,to_engine,ENGINE_USER):
    watch_list_rating_sorted = watch_list_rating.copy().sort_values(by=['rating'],ascending=[False])
    num_user = watch_list_rating_sorted.user_id.nunique()

    # get user lists
    user_list = read_user_list()
    similar_document_to_put = []
    similar_document_to_index = []
    print('Updating similar titles lists')
    counter_content = 0
    for user_id in watch_list_rating_sorted.sort_values(by='user_id',ascending=True).user_id.unique():
        print(f'{counter_content}/{num_user}',end='\r')
        watch_list_rating_sorted_user = watch_list_rating_sorted[watch_list_rating_sorted.user_id == user_id]
        num_play = min(len(watch_list_rating_sorted_user),5)
        watch_list_rating_sorted_user = watch_list_rating_sorted_user.head(num_play)
        topn_titles = watch_list_rating_sorted_user.media_id.tolist()
        topn_ratings = watch_list_rating_sorted_user.rating.tolist()
        titles_selected = []
        while len(titles_selected) < min(2,len(topn_titles)):
            titles_selected += random.choices(topn_titles,weights = topn_ratings,k=1)
            titles_selected = list(set(titles_selected))
        sim_titles = {"similarity_0":'',
                        "similarity_0_titles":'',
                        "similarity_1":'',
                        "similarity_1_titles":''}
        for i,title in enumerate(titles_selected):
            sim_titles[f'similarity_{i}'] = title
            sim_titles[f'similarity_{i}_titles'] = df_score_byid.sort_values(by=title,ascending=False).head(15)[title].sample(n=10).index.tolist()
        
        if user_id in user_list:
            similar_document_to_put.append({
                "id":user_id,
                "similarity_0":sim_titles['similarity_0'],
                "similarity_0_titles":sim_titles['similarity_0_titles'],
                "similarity_1":sim_titles['similarity_1'],
                "similarity_1_titles":sim_titles['similarity_1_titles'],
                "update":0
            })
        else:
            similar_document_to_index.append({
                "id":user_id,
                "similarity_0":sim_titles['similarity_0'],
                "similarity_0_titles":sim_titles['similarity_0_titles'],
                "similarity_1":sim_titles['similarity_1'],
                "similarity_1_titles":sim_titles['similarity_1_titles'],
                "update":0
            })
        # Request exceeds maximum allowed limit of 100 documents, send and reset similar variable
        if len(similar_document_to_put) == 50:
            print(f'{counter_content}: putting documents')
            resp = to_engine(engine_name=ENGINE_USER,documents=similar_document_to_put,mode='put')
            print(resp)
            similar_document_to_put = []
        if len(similar_document_to_index) == 50:
            print(f'{counter_content}: indexing documents')
            resp = to_engine(engine_name=ENGINE_USER,documents=similar_document_to_index,mode='index')
            print(resp)
            similar_document_to_index = []
        counter_content+=1
    if len(similar_document_to_put) != 0:
        resp = to_engine(engine_name=ENGINE_USER,documents=similar_document_to_put,mode='put')
        print(resp)
    if len(similar_document_to_index) != 0:
        resp = to_engine(engine_name=ENGINE_USER,documents=similar_document_to_index,mode='index')
        print(resp)
    return counter_content


# Personal


def infer_ratings(row):
    decay_rate = 0.99
    # series_movie = df_media[df_media.id == row['media_id']] 
    if row['completed'] == True:
        base_rate = 5.0
    else:
        # check length
        try:
            percent_completed = row['last_play_time_in_seconds'] / (int(row['meta_length'])*60)
        except:
            percent_completed = row['last_play_time_in_seconds'] / (120*60)
        base_rate = percent_completed * 5

    if row['updated_date'] == None:
        date_diff = date.today() - row['created_date']
    else:
        date_diff = date.today() - row['updated_date']
    scaled_rate = base_rate * decay_rate ** date_diff.days
    if row['completed'] == True:
        scaled_rate = max(scaled_rate,3.6)
    scaled_rate = max(scaled_rate,0.5)
    scaled_rate = min(scaled_rate,5.0)
    # print(scaled_rate)
    return scaled_rate


def run_recommendation(watch_list_rating,user,movie_mapping):
    print(f'Recommendating for user {user}')
    # load data
    reader = Reader(rating_scale=(0.5,5))
    data = Dataset.load_from_df(watch_list_rating,reader)
    # train model
    algo = SVD()
    algo.fit(data.build_full_trainset())
    # identify watched/unwatched videos
    unique_ids = watch_list_rating['media_id'].unique()
    rated_ids = watch_list_rating.loc[watch_list_rating['user_id']==user,'media_id']
    movies_to_predict = np.setdiff1d(unique_ids,rated_ids)

    # print(f'user {user} watched {len(rated_ids)} videos')

    recs = []
    for iid in movies_to_predict:
        recs.append((iid,algo.predict(uid=user,iid=iid).est))

    user_prediction = pd.DataFrame(recs,columns=['iid','predictions']).sort_values('predictions',ascending=False)
    user_prediction['title'] = user_prediction.iid.apply(lambda x: movie_mapping[x])
    # recommended_videos = user_prediction[user_prediction.title != None].head(10).title.tolist()
    recommended_videos = user_prediction[user_prediction.title != None].head(10).iid.tolist()
    return recommended_videos


def getPopularityRanks(df_movies_genres):
    ratings = defaultdict(int)
    for i,row in df_movies_genres.iterrows():
        ratings[row['media_id']] += 1
    rankings = defaultdict(int)
    rank = 1
    for media_id,ratingCount  in sorted(ratings.items(),key=lambda x:x[1],reverse=True):
        rankings[media_id] = rank
        rank +=1
    return dict(rankings)


def check_list(check,item,list_of_items):
    # print(genres, list_of_genres)
    if check == 'genres':
        try:
            if item in list_of_items:
                return True
            else:
                return False
        except:
            return False
    elif check =='watched':
        try:
            if item in list_of_items:
                return False
            else:
                return True
        except:
            return True


def run_recommendation_genres(df_media,df_movies_genres,user,rankings):
    def assign_rank(media_id):
        try:
            return rankings[media_id]
        except:
            return 9999
    # user = '73'
    # print(user)
    df_user_genres = df_movies_genres[df_movies_genres.user_id == user]
    watched_id = df_user_genres.media_id.tolist()
    df_user_genres = df_user_genres[df_user_genres.rating >=3.5]
    # df_user_genres.to_csv('test.csv')
    user_genres = dict(Counter(df_user_genres.genres.sum()))
    user_genres = sorted(user_genres.items(),key=lambda item: item[1],reverse=True)[0:min(10,len(user_genres))]
    user_genres = random.sample(list({k:v for k,v in user_genres}),5)
    user_recommendations_genres = {}
    for i,genres in enumerate(user_genres):
        user_recommendations_genres[f'genres_{i}']=genres
        df_movies_filtered_by_genres = df_media[['id','title','genres']].copy()
        df_movies_filtered_by_genres = df_movies_filtered_by_genres[df_movies_filtered_by_genres.id.apply(lambda x: check_list('watched',x,watched_id))]
        df_movies_filtered_by_genres = df_movies_filtered_by_genres[df_movies_filtered_by_genres.genres.apply(lambda x: check_list('genres',genres,x))]
        df_movies_filtered_by_genres['rank'] = df_movies_filtered_by_genres.id.apply(assign_rank)
        df_movies_filtered_by_genres.sort_values(by='rank',inplace=True,ignore_index=True)
        try:
            df_movies_filtered_by_genres = df_movies_filtered_by_genres.head(15).sample(10)
        except ValueError:
            df_movies_filtered_by_genres = df_movies_filtered_by_genres.head(15).sample(10,replace=True)
        # user_recommendations_genres[f'genres_{i}_recs'] = df_movies_filtered_by_genres.title.tolist()
        user_recommendations_genres[f'genres_{i}_recs'] = df_movies_filtered_by_genres.id.tolist()
        # break
    return user_recommendations_genres