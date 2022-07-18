import json, ast, math, time, uuid, os, datetime
import pandas as pd
from elastic_enterprise_search import AppSearch
import psycopg2
import psycopg2.extras
psycopg2.extras.register_uuid()

from model import content_based_filtering


# APPSEARCH

ENGINE_MEDIA = "media"
ENGINE_USER = "user"

with open('/Users/sam/Developer/credentials/app_search.json') as f:
    app_search_credentials = json.load(f)

endpoint = app_search_credentials['end-point']+':9243'
private_key = app_search_credentials['private-key']

app_search = AppSearch(
    endpoint,
    bearer_auth=private_key
)


def list_engine():
    resp = app_search.list_engines()
    return resp['results']


def read_media_cache():
    print('Reading cached media data')
    df_media = pd.read_csv('cached_data/cached_media.csv')
    print('Preprocessing cached media data')
    df_media.tags = df_media.tags.apply(lambda x: ast.literal_eval(x))
    df_media.genres = df_media.genres.apply(lambda x: ast.literal_eval(x))
    df_media.directors = df_media.directors.apply(lambda x: ast.literal_eval(x))
    df_media.cast = df_media.cast.apply(lambda x: ast.literal_eval(x))
    df_media.duration = df_media.duration.astype('str')
    df_media.released_year = df_media.released_year.astype('str')
    df_media.added_year = df_media.added_year.astype('str')
    df_media.previews = df_media.previews.apply(lambda x: ast.literal_eval(x))
    df_media.series_meta_season_id = df_media.series_meta_season_id.fillna('0')
    df_media.series_meta_season_id = df_media.series_meta_season_id.apply(lambda x: str(int(x)))
    df_media.series_meta_season_id = df_media.series_meta_season_id.astype('str')
    df_media.series_meta_episode_id = df_media.series_meta_episode_id.fillna('0')
    df_media.series_meta_episode_id = df_media.series_meta_episode_id.apply(lambda x: str(int(x)))
    df_media.series_meta_episode_id = df_media.series_meta_episode_id.astype('str')
    # df_media.countries_allowed = df_media.countries_allowed.apply(lambda x: ast.literal_eval(x))
    # df_media.countries_disallowed = df_media.countries_disallowed.apply(lambda x: ast.literal_eval(x))
    df_media.liked = df_media.liked.astype('str')
    df_media.unliked = df_media.unliked.astype('str')
    df_media.played = df_media.played.astype('str')
    print('Read cached media data')
    return df_media


def read_media_engine():
    print('Reading media data from AppSearch')
    resp = app_search.list_engines()['results']
    for r in resp:
        if r['name'] == ENGINE_MEDIA:
            num_video = r['document_count']
    page_size = math.ceil(num_video/1000)
    print(f'page_size: {page_size}')
    media_as = []
    # Loop through the pages, query result size cap at 1000
    for p in range(page_size):
        resp = app_search.search(engine_name=ENGINE_MEDIA,query="",current_page=p+1,page_size=1000)
        media_as = media_as + resp['results']
    print(f'collected {len(media_as)} results')
    # Check for duplication
    if len(media_as) != len(list(set([r['id']['raw'] for r in media_as]))):
        print("Error: There are duplications in the extracted results!")
    
    # Covert data to DataFrame
    df_media_columns= ["id","title","description","type","language","tags","genres","directors","cast","duration","released_date","released_year","added_date","added_year","urls","previews","title_image","thumbnail_image","preview_image","featured_image","series_meta_series_id","series_meta_seasons_count","series_meta_season_id","series_meta_episode_id","collections_meta_collection_id","collections_meta_collection_order_id","max_quality","maturity_rating","featured","featured_from","featured_to","countries_allowed","countries_disallowed","licence_termination","licence_renewal","licence_renewal_reminder","play_next_after","liked","unliked","played"]
    df_media = pd.DataFrame(columns=df_media_columns)

    for record in media_as:
        record_new = {}
        for key in record.keys():
            if key != '_meta':
                record_new[key] = record[key]['raw']
        df_media = df_media.append(record_new,ignore_index=True)
    
    # Cache media
    df_media.to_csv('cached_data/cached_media.csv',index=False)
    print('Cached data extracted from AppSearch')
    return df_media


def read_user_list():
    print('Read user list from AppSearch')
    resp = app_search.list_engines()['results']
    for r in resp:
        if r['name'] == ENGINE_USER:
            num_video = r['document_count']
    page_size = math.ceil(num_video/1000)
    print(f'page_size: {page_size}')
    user_as = []
    for p in range(page_size):
        resp = app_search.search(engine_name=ENGINE_USER,query="",current_page=p+1,page_size=1000)
        user_as = user_as + resp['results']
    user_list = [user['id']['raw'] for user in user_as]
    print(f'There are {len(user_list)} user records')
    return user_list


def read_user_update():
    print('Read user data from AppSearch')
    resp = app_search.list_engines()['results']
    for r in resp:
        if r['name'] == ENGINE_USER:
            num_video = r['document_count']
    page_size = math.ceil(num_video/1000)
    print(f'page_size: {page_size}')
    user_as = []
    for p in range(page_size):
        resp = app_search.search(engine_name=ENGINE_USER,query="",current_page=p+1,page_size=1000)
        user_as = user_as + resp['results']
    user_to_update = []
    for user in user_as:
        if user['unique_titles']['raw'] >= 5 and user['update']['raw'] == "1":
            user_to_update.append(user['id']['raw'])
    print(f'Extracted: {len(user_as)} users\nUser to update: {len(user_to_update)} users')
    return user_to_update


def read_user_current_recommendations(df_media,id):
    print(f"Get user {id}' current recommmendation list")
    current_recommendations = []
    user_status = 'new'
    try:
        current_recommendations = app_search.get_documents(
            engine_name=ENGINE_USER,
            document_ids=[id]
        )[0]['recommendations']
        print(f'User {id} exist, have personalised recommendation already')
        user_status = 'exist'
    except TypeError: # if user does not exist in elasticsearch
        print(f'User {id} is a new user')
        current_recommendations = df_media.id.sample(n=10).to_list()
    except KeyError: # if user does not have personalised recommendations yet
        print(f'User {id} exist, but does not have personalised recommendations')
        current_recommendations = df_media.id.sample(n=10).to_list()
        user_status = 'exist'
    return current_recommendations, user_status


def add_user_count(id):
    resp = app_search.get_documents(engine_name=ENGINE_USER,document_ids=id)
    print(resp)
    try:
        unique_count = int(resp[0]['unique_titles']) + 1
    except TypeError:
        unique_count = 1
    app_search.put_documents(engine_name=ENGINE_USER,documents=[{'id':id,'unique_titles':unique_count}])
    return None


def to_engine(engine_name, documents, mode):
    if mode == 'index':
        for i in range(20):
            try:
                resp = app_search.index_documents(engine_name=engine_name,documents=documents)
                break
            except:
                pass
            print(f'tries {i+1}: failed indexing documents to {engine_name}')
            if i == 19:
                print(documents)
            time.sleep(2)
    elif mode == 'put':
        for i in range(20):
            try:
                resp = app_search.put_documents(engine_name=engine_name,documents=documents)
                break
            except:
                pass
            print(f'tries {i+1}: failed putting documents to {engine_name}')
            if i == 19:
                print(documents)
            time.sleep(2)
    else:
        pass
    return resp


# POSTGRESQL

HOST = 'localhost'
DATABASE = 'database'
USER = 'postgres'
PASSWORD = 'postgres'
PORT= '5432'
TABLE_play = 'play_history'
TABLE_rate = 'rate_history'


def read_table_records():
    print('Connecting to PostgreSQL')
    con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    cursor = con.cursor()
    query = f'''select * from pg_catalog.pg_tables where schemaname != 'pg_catalog' and schemaname != 'information_schema';'''
    cursor.execute(query)
    table_records = {}
    for schemaname, tablename, tableowner, tablespace, hasindexes, hasrules, hastriggers, rowsecurity in cursor.fetchall():
        query = f'''select count(*) from {tablename}'''
        cursor.execute(query)
        table_records[tablename] = cursor.fetchall()[0][0]
    con.close()
    print('Extracted table counts')
    return table_records


def read_table_all(table):
    print(f'Connecting to PostgreSQL: {table}')
    con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    query = f'''select * from {table}'''
    df = pd.read_sql_query(query,con=con)
    print(f'Extracted {len(df)} records from {table} as DataFrame')
    df.to_csv(f'cached_data/cached_{table}.csv',index=False)
    print(f'Cached {table} data')
    return df


def insert_table_new_activity(user_activity):
    print(f'Send new activity to PostgreSQL: {TABLE_play}')
    con = psycopg2.connect(host = HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
    cursor = con.cursor()

    query = f"""
    insert into {TABLE_play} (
        user_id,
        media_id,
        last_play_time_in_seconds,
        completed,continue_playing_active,
        history_record_active,
        created_date,
        created_ts)
    values (
        '{user_activity['user_id']}',
        '{user_activity['media_id']}',
        {user_activity['last_play_time_in_seconds']},
        {user_activity['completed']},
        {user_activity['continue_playing_active']},
        {user_activity['history_record_active']},
        TO_DATE('{user_activity['created_date']}','YYYY-MM-DD'),
        '{datetime.datetime.fromtimestamp(user_activity['created_ts']).strftime('%Y-%m-%d %H:%M:%S')}');
    """
    cursor.execute(query)
    con.commit()
    # break
    cursor.close()
    con.close()
    return None
    

# Similar Score

def read_score_similarity(media_titles,df_media):
    file_cached_sim_score = 'cached_data/cached_similarity_score_byid.csv'

    if os.path.exists(file_cached_sim_score):
        print('Cached file found, reading cached similarity score')
        df_score_cache = pd.read_csv(file_cached_sim_score,index_col=0)
        cached_titles = df_score_cache.index.tolist()
        cached_titles.sort()
        if media_titles == cached_titles:
            print('No new titles detected, using cached similar score')
            df_score_byid = df_score_cache
        else:
            print('New titles detected, calculating similarity score')
            df_score_byid,df_score_bytitle = content_based_filtering(df_media)
    else:
        print('Cached file not found, calculating similarity score')
        df_score_byid,df_score_bytitle = content_based_filtering(df_media)
    
    return df_score_byid