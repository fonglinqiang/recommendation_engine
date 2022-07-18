from flask import Flask, session, redirect, url_for, render_template
from functools import wraps
import json, math, random
import pandas as pd
from datetime import date,datetime
from elastic_enterprise_search import AppSearch
import psycopg2
import psycopg2.extras

import numpy as np

import time
import uuid
psycopg2.extras.register_uuid()

from data_management import ENGINE_MEDIA, ENGINE_USER
from data_management import list_engine, read_media_cache, read_media_engine, read_user_list, to_engine, read_user_update, read_user_current_recommendations, add_user_count
from data_management import HOST, DATABASE, USER, PASSWORD, PORT, TABLE_play
from data_management import read_table_records, read_table_all, insert_table_new_activity
from data_management import read_score_similarity

from model import preprocessing, update_content_recommendation, run_recommendation, run_recommendation_genres, infer_ratings, getPopularityRanks


app = Flask(__name__)
app.secret_key = 'your-secret-key'

with open('/Users/sam/Developer/credentials/app_search.json') as f:
    app_search_credentials = json.load(f)

endpoint = app_search_credentials['end-point']+':9243'
private_key = app_search_credentials['private-key']

app_search = AppSearch(
    endpoint,
    bearer_auth=private_key
)

# Get media data from cached or appsearch engine
try:
    df_media = read_media_cache()
except:
    print('Cached Media Data Unavailable')
    df_media = read_media_engine()
    
movie_mapping = df_media[['id','title']].set_index('id')['title'].to_dict()
df_media['title'] = df_media.id.apply(lambda x: movie_mapping[x])
df_movie_mapping = df_media[['id','title','type']].rename(columns={'id':'media_id'})


categories = [
    {
        'name': "Most Popular",
        'tags': ['meta_most_popular'],
        'genres': [],
        'order': 1
    },
    {
        'name': "Recently Added",
        'tags': ['meta_recently_added'],
        'genres': [],
        'order': 2
    },
    {
        'name': "Highly Rated",
        'tags': ['meta_highly_rated'],
        'genres': [],
        'order': 3
    },
]

# Start flask app

def api_key_required(foo):
    @wraps(foo)
    def wrap(*args, **kwargs):
        if 'user' in session:
            return foo(*args, **kwargs)
        else:
            print('out')
            return redirect(url_for('home'))
    return wrap


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/connect_appsearch')
def connect_appsearch():
    return render_template('connect_appsearch.html', con = list_engine())


@app.route('/connect_psql')
def connect_psql():
    return render_template('connect_psql.html',table_records = read_table_records())


@app.route('/get_videos')
def get_videos():
    df_media = read_media_engine()
    return render_template('get_data.html',statement=f"Extracted {len(df_media)} records from AppSearch {ENGINE_MEDIA} engine")


@app.route('/get_play_history')
def get_play_history():
    watch_list = read_table_all(TABLE_play)
    return render_template('get_data.html',statement=f"Extracted {len(watch_list)} records from Postgres {TABLE_play} table")


@app.route('/user', defaults = {'id':'1'}, methods = ['GET'])
@app.route('/user/<id>')
def user(id):
    static_recommendations = {}

    for cat in categories:
        if cat['order'] == 1:
            static_recommendations['position_1_title'] = cat['name']
            static_recommendations['position_1'] = df_media[(df_media.tags.apply(lambda x: cat['tags'][0] in x)&(df_media.type.isin(['movie','series'])))].head(10).id.to_list()
        elif cat['order'] == 2:
            static_recommendations['position_2_title'] = cat['name']
            static_recommendations['position_2'] = df_media[(df_media.tags.apply(lambda x: cat['tags'][0] in x)&(df_media.type.isin(['movie','series'])))].head(10).id.to_list()
        elif cat['order'] == 3:
            static_recommendations['position_3_title'] = cat['name']
            static_recommendations['position_3'] = df_media[(df_media.tags.apply(lambda x: cat['tags'][0] in x)&(df_media.type.isin(['movie','series'])))].head(10).id.to_list()
        else:
            pass

    resp = app_search.get_documents(
        engine_name=ENGINE_USER,
        document_ids=[id]
    )
    print(resp)
    user_info = resp[0]
    if user_info ==  None:
        user_not_found = True
    elif user_info['id'] != id:
        user_not_found = True
    else:
        user_not_found = False
    return render_template('user.html',static_recommendations = static_recommendations, user_data = user_info, error_msg = f'User {id} personalised recommendation is not available', movie_mapping = movie_mapping, user_not_found=user_not_found)


@app.route('/new_activity',defaults = {'id':'1'}, methods = ['GET'])
@app.route('/new_activity/<id>')
def new_activity(id):
    # Get current recommendations
    current_recommendations, user_status = read_user_current_recommendations(df_media,id)
    
    # only watching movies for simulation
    user_id_video_type = 'start'
    while user_id_video_type in ['series','episode','start']:
        user_id_video_name = random.choice(current_recommendations)
        user_id_video = df_media[df_media.id == user_id_video_name]
        user_id_video_type = user_id_video.type.to_list()[0]
        
    # get title duration and format data in dict
    user_id_video_index = user_id_video.index.tolist()[0]
    test_user_new_video_len = user_id_video.loc[user_id_video_index,'duration']
    test_user_new_video_len = test_user_new_video_len.split(':')
    test_user_new_video_len = float(test_user_new_video_len[0])*60 + float(test_user_new_video_len[1])
    user_activity = {
        'user_id': id, 
        'media_id': uuid.UUID(user_id_video.loc[user_id_video_index,'id']), 
        'last_play_time_in_seconds': test_user_new_video_len*0.98, 
        'completed': True, 
        'continue_playing_active': True,
        'history_record_active': True,
        'created_date': date.today().strftime("%Y-%m-%d"), 
        'created_ts': int(time.time()),
        'updated_date': math.nan,
        'updated_ts': math.nan 
    }
    print(f'New activity for user {id}: ', movie_mapping[user_id_video_name])
    print(user_activity)

    insert_table_new_activity(user_activity)
    add_user_count(id)

    print(f'Update new activity in user engine: {id}')

    if user_status == 'exist':
        to_engine(engine_name=ENGINE_USER,documents=[{"id":id,"update":1}],mode='put')
    elif user_status == 'new':
        to_engine(engine_name=ENGINE_USER,documents=[{"id":id,"update":0,"unique_titles":1}],mode='index')

    return render_template('new_activity.html',user_activity=user_activity,user_id_video_name=movie_mapping[user_id_video_name])


@app.route('/recommend')
def recommend():
    # Get users with new watch activities
    user_to_update = read_user_update()

    # Read play history data from PostgreSQL
    watch_list = read_table_all(TABLE_play)
    watch_list.media_id = watch_list.media_id.apply(str)

    # Preprocessing
    watch_list_rating = preprocessing(watch_list,df_media)

    df_movies_genres = watch_list_rating.merge(df_media[['id','genres']].rename(columns={'id':'media_id'}),on='media_id',how='left')
    rankings = getPopularityRanks(df_movies_genres)

    recommendation_document_to_put = []
    for i,u in enumerate(user_to_update):
        print(f'{i}/{len(user_to_update)}',end='\r')
        new_user_rec = {
            "id": u,
            "updated":0,
            }
        new_user_rec["recommendations"] = run_recommendation(watch_list_rating,u,movie_mapping)
        new_user_rec = dict(list(new_user_rec.items())+ list(run_recommendation_genres(df_media,df_movies_genres,u,rankings).items()))
        # update user's recommendations in appsearch
        recommendation_document_to_put.append({
            "id":new_user_rec['id'],
            "update":new_user_rec['updated'],
            "recommendations":new_user_rec['recommendations'],
            "genres_0":new_user_rec['genres_0'],
            "genres_0_recs":new_user_rec['genres_0_recs'],
            "genres_1":new_user_rec['genres_1'],
            "genres_1_recs":new_user_rec['genres_1_recs'],
            "genres_2":new_user_rec['genres_2'],
            "genres_2_recs":new_user_rec['genres_2_recs'],
            "genres_3":new_user_rec['genres_3'],
            "genres_3_recs":new_user_rec['genres_3_recs'],
            "genres_4":new_user_rec['genres_4'],
            "genres_4_recs":new_user_rec['genres_4_recs'],
        })
        if len(recommendation_document_to_put) == 50:
            resp = to_engine(engine_name=ENGINE_USER,documents=recommendation_document_to_put,mode='put')
            print(resp)
            recommendation_document_to_put = []
    if len(recommendation_document_to_put) != 0:
        resp = to_engine(engine_name=ENGINE_USER,documents=recommendation_document_to_put,mode='put')
        print(resp)
    return render_template('recommend.html',users = user_to_update,num_user = len(user_to_update))


@app.route('/similar')
def similar():
    # Read play history data from PostgreSQL
    watch_list = read_table_all(TABLE_play)
    watch_list.media_id = watch_list.media_id.apply(str)

    # Preprocessing
    watch_list_rating = preprocessing(watch_list,df_media)

    # Content-based recommendations
    media_titles = df_media[df_media.type != 'episode'].id.tolist()
    media_titles.sort()

    # read cached similarity scores
    df_score_byid = read_score_similarity(media_titles,df_media)

    # update similar recommendations
    counter_content = update_content_recommendation(watch_list_rating,read_user_list,df_score_byid,to_engine,ENGINE_USER)

    return render_template('similar.html',counter_content=counter_content)
