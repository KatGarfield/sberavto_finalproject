"""The purpose of this module is to define functions for later import.
No need to run this file.

This module defines functions that will extract target variable from ga_hits file
and merge them with ge_sessions dataset.
Also, it defines functions that will be used to create two new dataframes.

One dataframe df_prev will contain additional data extracted from ga_hits file such as
session duration and clients' previous activity counts.
At pipeline preprocessing stage these new data will be merged with each client's
next session if available (see pipeline.py).

The other new dataframe df_reference will contain lists of categorical features unique values
and their probabilities of target actions.

Functions defined below will be used to preprocess original ga_sessions df:
replace outliers, fill nas, create regional affiliation feature,
group utm features into regions and major marketing campaigns and then calculate
success probabilities within these groups,
transform cities and device_screen_resolutions into success probabilities,
drop columns and duplicate rows that won't be used as reference.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def isoutlier_3sigma(df_column):
    '''
    Takes column from a dataframe and returns threshold values
    defined as 3 standard deviations below and above mean value
    '''

    low_threshold = df_column.mean() - 3 * df_column.std()
    high_threshold = df_column.mean() + 3 * df_column.std()
    return low_threshold, high_threshold


def add_target_column():
    '''
    Reads ga_hits and gs_session csv-files.

    In ga_hits:
     a) calculates each session duration as a difference between first
        and last event action timestamp ('session_duration' column)
     b) calculates total number of actions for each session_id ('event_action' column)
     c) calculates number of target actions for each session_id ('result' column)

    Then merges these new data with ga_sessions on session_id column.
    Also creates target column based on 0 or more than 0 target actions.
    Replaces outliers in three of the new columns with maximum value.
    Low threshold is ignored for the lack of values below zero.

    Returns ga_sessions merged with 4 new columns including target column.
    '''

    df_hits = pd.read_csv(
        r'..\data\ga_hits.csv', dtype={'session_id': 'object'})
    df_sessions = pd.read_csv(
        r'..\data\ga_sessions.csv',
        dtype={'session_id': 'object', 'client_id': 'object',
               'visit_date': 'object', 'visit_time': 'object'})

    # Calculate duration of each session

    df_time = df_hits[['session_id', 'hit_time']].groupby('session_id').agg({
        'hit_time': ('min', 'max')}).reset_index()
    df_time['session_duration'] = df_time[('hit_time', 'max')] - df_time[('hit_time', 'min')]
    df_time.drop(columns=['hit_time'], inplace=True)
    df_time.columns = df_time.columns.droplevel(1)

    # Create dataframe with target action, total number of actions and
    # duration for each session from df_hits

    df_hits['result'] = df_hits['event_action'].apply(
        lambda x: 1 if x in [
            'sub_car_claim_click',
            'sub_car_claim_submit_click',
            'sub_open_dialog_click',
            'sub_custom_question_submit_click',
            'sub_call_number_click',
            'sub_callback_submit_click',
            'sub_submit_success',
            'sub_car_request_submit_click',
        ] else 0
    )
    df_to_merge = df_hits[['session_id', 'result', 'event_action']].groupby(
        'session_id').agg({'result': 'sum', 'event_action': 'count'}).reset_index()
    df_to_merge = pd.merge(df_to_merge, df_time, how='left', on='session_id')

    # Add df_hits columns and target value to df_sessions

    df = pd.merge(df_sessions, df_to_merge, how='left', on='session_id')
    df.dropna(subset=['result'], inplace=True)
    df['target'] = df['result'].apply(lambda x: 1 if x else 0)

    # Replace outliers in new columns with max values before creating previous
    # behaviour dataframe

    low_threshold, high_threshold = isoutlier_3sigma(df.result)
    df['result'] = df['result'].apply(
        lambda x: x if x < high_threshold else high_threshold)

    low_threshold, high_threshold = isoutlier_3sigma(df.event_action)
    df['event_action'] = df['event_action'].apply(
        lambda x: x if x < high_threshold else high_threshold)

    low_threshold, high_threshold = isoutlier_3sigma(df.session_duration)
    df['session_duration'] = df['session_duration'].apply(
        lambda x: x if x < high_threshold else high_threshold)

    return df


def create_previous_activity_df(df):
    '''
    Creates a new dataframe with client activity history based on data
    received from ga_hits.
    Step 1:
    For each session_id aggregates 'result' and 'event_action' counts for
    current session and 10 or less previous visits based on client_id and visit_number.
    Also adds session_duration and visit_datetime columns.
    Step 2:
    Calculates mean values per session_id for all 'result' columns
    and 'all event_action' columns. Drops separate visits result and event action columns.
    Step 3:
    Visit number is offset by one.
    This is necessary so that the new df could be merged on client_id and visit_number
    with the original df without any data leak.
    (e.g. client_id in the original df with visit number 3 will be merged with
    the same client_id activity data from visit numbers 1 and 2).

    Saves new df with four previous activity features to 'data/' folder.
    Returns df with four previous activity features.
    '''

    df_prev = df[['client_id', 'visit_number', 'visit_date', 'session_duration']]
    df_prev.visit_date = df_prev.visit_date.astype('datetime64[ns]')
    df_prev.drop_duplicates(subset=['client_id', 'visit_number'], inplace=True)
    new_feats = [[], []]
    for num in range(10):
        new_feats[0].append('result' + str(num + 1))
        new_feats[1].append('action' + str(num + 1))

        # Rename columns to avoid duplicate names at merging stage

        df_prev_tmp = df[['client_id', 'visit_number', 'event_action', 'result']].rename(
            columns={'result': new_feats[0][-1], 'event_action': new_feats[1][-1]})
        df_prev_tmp.drop_duplicates(subset=['client_id', 'visit_number'], inplace=True)
        df_prev_tmp['visit_number'] = df_prev_tmp['visit_number'] + num
        df_prev = pd.merge(df_prev, df_prev_tmp, how='left', on=['client_id', 'visit_number'])

    # Calculate mean values based on existing data

    df_prev['mean_result'] = df_prev[new_feats[0]].mean(axis=1)
    df_prev['mean_actions'] = df_prev[new_feats[1]].mean(axis=1)

    # Drop all earlier results, keep mean values

    df_prev.drop(columns=new_feats[0] + new_feats[1], inplace=True)

    # Offset visit_number by one to avoid data leak at merging step

    df_prev['visit_number'] = df_prev['visit_number'] + 1
    df_prev.rename(columns={'visit_date': 'prev_visit',
                            'session_duration': 'prev_session_duration'}, inplace=True)
    df_prev.to_csv(r'..\data\df_previous_activity.csv', index=False)

    return df_prev


def add_prev_activity(df):
    '''
    Merges original dataframe with previous activity dataframe created earlier
    on client_id and visit_number columns,
    then offsets visit number 5 times to fill some of the NA values for visits
    skipped in the original df (and also missing in the previous activity df)

    Calculates days since client's previous visit as a difference between current visit_date
    and visit_date from previous activity df.

    Fills all NAs for first time clients with no history and for returning clients with missing
    previous data with columns median values.

    Returns modified original df
    '''

    df_prev = create_previous_activity_df(df)

    # Copy df for multiple merge

    df_raw = df.copy()
    df = pd.merge(df, df_prev, on=['client_id', 'visit_number'], how='left')

    for num in range(6):
        df_prev['visit_number'] = df_prev['visit_number'] + 1
        df2 = pd.merge(df_raw, df_prev, on=['client_id', 'visit_number'], how='left')
        df.update(df2, overwrite=False)

    df['days_after_last_visit'] = ((df['visit_date'] - df['prev_visit']) /
                                   np.timedelta64(1, 'D'))

    # Fill missing values and first time client values in the new
    # columns with median value

    new_cols = ['mean_result', 'mean_actions', 'days_after_last_visit',
                'prev_visit', 'prev_session_duration']
    for col in new_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def fill_nas(df):
    '''
    Fills NAs in device_brand column with 'Apple' if device browser is Safari
    Fills NAs in columns with numeric datatype with column median value
    Fills NAs in columns with object datatype with '(not set)'
    '''

    def fill_brand_nas(df):
        for index, row in df[df.device_brand.isna()].iterrows():
            if row['device_browser'] == 'Safari':
                df.loc[index, 'device_brand'] = 'Apple'
        return df

    df = fill_brand_nas(df)

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_features:
        df[column] = df[column].fillna(df[column].median())

    df.visit_date = df.visit_date.fillna(df.visit_date.mode())
    df.visit_date = df.visit_date.astype('datetime64[ns]')

    categorical_features = list(df.select_dtypes(include=['object']).columns)
    for column in categorical_features:
        df[column] = df[column].fillna('(not set)')

    return df


def add_ad_startdates(df):
    '''
     Creates time related new features

     a) Calculates how long ago the ad was launched
        (number of days is defined as a difference between current session date
        and the minimum date found in the dataframe for this utm_adcontent name)
    b) Adds new date column based on visit_date and transforms it into numeric as follows:
       year multiplied by 365 days + month multiplied by 30.5 days + day number
    c) Bring date and time columns to numeric type

    Returns modified df
    '''

    # For each advertisement calculate number of days since its publication

    ad_start_dates = {}
    for ad in df.utm_adcontent.unique():
        ad_start = df[df.utm_adcontent == ad].visit_date.min()
        ad_start_dates[ad] = ad_start

    df['ad_start_tmp'] = df.utm_adcontent.apply(lambda x: ad_start_dates.get(x))

    return df


def add_region(df):
    '''
    Based on external data defines cities coordinates and respective regions.
    Regions are defined based on the following criteria:
    a) cities with over 500 successful sessions counted as separate regions
    b) foreign countries and Russian administrative regions with over 7000 sessions
       kept as separate regions
    c) Russian towns not found in the available geodata dictionary joined into 'Russia'
    d) Russian regions with less than 7000 sessions replaced with federal subjects
    e) foreign countries with less than 7000 sessions replaced with 'foreign region'
    f) '(not set)' kept as is

    Also defines capital cities of administrative regions and foreign countries and calculates
    distance to capital based on latitude and longitude.

    Returns modified df with new columns: 'region' and 'dist_to_capital'
    '''

    def create_cities_dict():
        '''
        Creates dictionary with the following structure:
            {admin_name:
             {city1: str(city1 region),
              city2: str(city2 region)
              ...},
             country:
             {city1: str(city1 country),
              city2: str(city2 country),
              ...},
        }
        '''

        # Import global cities dataset with cities coordinates, region and country affiliation

        df_cities = pd.read_csv(r'..\data\worldcities.csv')
        df_cities = df_cities[['city_ascii', 'admin_name', 'country']]

        # Transform data to dictionary

        df_cities.set_index('city_ascii', inplace=True)

        return df_cities.to_dict()

    cities_admins = create_cities_dict()

    # Create new columns for region (in Russia only)

    df['region'] = df.geo_city.apply(lambda x: None if cities_admins['country'].get(
        x, 'other') != 'Russia' else cities_admins['admin_name'].get(x, None))

    # Update region names:
    # for foreign countries use country as region name
    # for Russia use region (admin_name) added above

    df['geo_country'] = df['geo_country'].apply(lambda x: None if x == 'Russia' else x)
    df['region'] = df.region.combine_first(df.geo_country)
    df['region'] = df['region'].fillna('Russia')

    # Create federal districts dictionary to merge small regions into bigger entities

    federal_districts = {}

    for region in list(df[df.geo_country != 'Russia'][
                           'geo_country'].unique()):
        federal_districts[region] = 'foreign region'

    for region in ['Chechnya', 'Stavropol’skiy Kray', 'Karachayevo-Cherkesiya',
                   'Kabardino-Balkariya', 'Ingushetiya', 'Dagestan', 'North Ossetia']:
        federal_districts[region] = 'North Caucasian'

    for region in ['Moskovskaya Oblast’', 'Yaroslavskaya Oblast’', 'Tul’skaya Oblast’',
                   'Lipetskaya Oblast’', 'Ryazanskaya Oblast’', 'Kaluzhskaya Oblast’',
                   'Tambovskaya Oblast’', 'Tverskaya Oblast’', 'Kostromskaya Oblast’',
                   'Kurskaya Oblast’', 'Voronezhskaya Oblast’', 'Belgorodskaya Oblast’',
                   'Smolenskaya Oblast’', 'Orlovskaya Oblast’', 'Bryanskaya Oblast’',
                   'Vladimirskaya Oblast’', 'Ivanovskaya Oblast’']:
        federal_districts[region] = 'Central'

    for region in ['Leningradskaya Oblast’', 'Arkhangel’skaya Oblast’', 'Komi',
                   'Vologodskaya Oblast’', 'Kareliya', 'Novgorodskaya Oblast’',
                   'Kaliningradskaya Oblast’', 'Pskovskaya Oblast’', 'Murmanskaya Oblast’']:
        federal_districts[region] = 'Northwestern'

    for region in ['Tatarstan', 'Ul’yanovskaya Oblast’', 'Samarskaya Oblast’',
                   'Mariy-El', 'Nizhegorodskaya Oblast’', 'Bashkortostan',
                   'Saratovskaya Oblast’', 'Orenburgskaya Oblast’', 'Kirovskaya Oblast’',
                   'Udmurtiya', 'Permskiy Kray', 'Chuvashiya', 'Penzenskaya Oblast’',
                   'Mordoviya']:
        federal_districts[region] = 'Volga'

    for region in ['Krasnodarskiy Kray', 'Astrakhanskaya Oblast’', 'Rostovskaya Oblast’',
                   'Rostovskaya Oblast’', 'Volgogradskaya Oblast’', 'UkraineCrimea',
                   'Adygeya', 'Kalmykiya']:
        federal_districts[region] = 'Southern'

    for region in ['Chelyabinskaya Oblast’', 'Sverdlovskaya Oblast’', 'Tyumenskaya Oblast’',
                   'Khanty-Mansiyskiy Avtonomnyy Okrug-Yugra', 'Yamalo-Nenetskiy Avtonomnyy Okrug',
                   'Kurganskaya Oblast’']:
        federal_districts[region] = 'Ural'

    for region in ['Krasnoyarskiy Kray', 'Tyva', 'Omskaya Oblast’', 'Kemerovskaya Oblast’',
                   'Novosibirskaya Oblast’', 'Altayskiy Kray', 'Tomskaya Oblast’',
                   'Irkutskaya Oblast’', 'Khakasiya', 'Altay']:
        federal_districts[region] = 'Siberian'

    for region in ['Primorskiy Kray', 'Khabarovskiy Kray', 'Zabaykal’skiy Kray',
                   'Amurskaya Oblast’', 'Sakha (Yakutiya)', 'Sakhalinskaya Oblast’',
                   'Buryatiya', 'Kamchatskiy Kray', 'Magadanskaya Oblast’',
                   'Yevreyskaya Avtonomnaya Oblast’', 'Chukotskiy Avtonomnyy Okrug']:
        federal_districts[region] = 'Far Eastern'

    # Single out major cities with most successful sessions as separate regions

    df_positive = df[df.target == 1]
    major_cities = list(df_positive.geo_city.value_counts()[
                            df_positive.geo_city.value_counts() > 1000].index)
    if '(not set)' in major_cities:
        major_cities.remove('(not set)')
    for city in major_cities:
        df.loc[df.geo_city == city, 'region'] = city

    # Make a list of small regions with number of sessions below 7000

    small_regions = []
    for region in df.region.unique():
        if len(df[df.region == region]) < 10000:
            small_regions.append(region)

    # Apply dictionary to small regions, then replace empty line  with 'Russia'
    # (empty string is for small russian cities with no region found in cities_coords dictionary)

    df.region = df.region.apply(
        lambda x: x if x not in small_regions else federal_districts.get(x, 'Russia'))

    return df


def modify_device_city(df):
    '''
    Joins device screen resolution and device brand into one column that supposedly
    acts as a replacement for a more valuable but mostly empty device model column

    Transforms device screen resolution and geo city into success probabilities

    Return modified df
    '''

    # Join brand and screen resolution as a replacement for device brand & model

    df.device_screen_resolution = (df.device_screen_resolution +
                                   df.device_brand)
    resolution_success = df[['device_screen_resolution', 'session_id',
                             'target']].groupby('device_screen_resolution').agg(
        {'session_id': 'count', 'target': 'sum'}).reset_index()
    resolution_success['device_screen_resolution_success'] = round(
        resolution_success['target'] / resolution_success['session_id'], 3)
    resolution_success = resolution_success[resolution_success.session_id > 50]
    resolution_success.drop(columns=['session_id', 'target'], inplace=True)
    df = pd.merge(df, resolution_success, how='left', on=['device_screen_resolution'])

    city_success = df[['geo_city', 'session_id', 'target']].groupby('geo_city').agg(
        {'session_id': 'count', 'target': 'sum'}).reset_index()
    city_success['geo_city_success'] = round(city_success['target'] /
                                             city_success['session_id'], 3)
    city_success = city_success[city_success.session_id > 30]
    city_success.drop(columns=['session_id', 'target'], inplace=True)
    df = pd.merge(df, city_success, how='left', on=['geo_city'])

    return df


def drop(df):
    '''
    Drops useless columns
    Returns modified df
    '''

    cols_to_drop = [
        'session_id',
        'client_id',
        'visit_time',
        'visit_number',
        'visit_date',
        'prev_visit',
        'days_after_last_visit',
        'device_brand',
        'device_category',
        'device_browser',
        'device_model',
        'device_os',
        'geo_country',
        'utm_medium',
        'session_duration',
        'result',
        'event_action',
        'mean_result',
        'mean_actions',
        'prev_session_duration',
        'target',
    ]

    df.drop(columns=cols_to_drop, inplace=True)
    df.drop_duplicates(subset=['utm_source', 'utm_adcontent', 'utm_keyword','utm_campaign',
                               'device_screen_resolution', 'geo_city', 'region'], inplace=True)

    return df


def transform_cat_feats(df):
    '''
    Makes a list of most important utm campaigns.
    Adds major campaign name to other utm columns.
    Then joins all categorical columns except geo_city and device_screen_resolution
    with respective region.
    Calculates success probability per value in each categorical column.
    Adds success probabilities as separate columns.
    Adding region and major campaign helps define success probability more precisely for
    bigger groups that may have very different priorities.
    '''

    coef_cols = ['utm_medium', 'utm_campaign', 'utm_source', 'utm_adcontent', 'utm_keyword']
    major_campaigns = list(df[df.target == 1].utm_source.value_counts().head(4).index)
    dependency_cols = ['utm_source', 'utm_keyword', 'utm_adcontent']
    df['major_campaign'] = df.utm_campaign.apply(lambda x: x[:5] if x in major_campaigns else '')
    for column in dependency_cols:
        df[column] = df['major_campaign'] + df[column]
    for column in coef_cols:
        tmp_success = df[['region', column, 'session_id', 'target']].groupby(['region', column]).agg(
                {'session_id': 'count', 'target': 'sum'}).reset_index()
        tmp_success[column + '_success'] = round(
            tmp_success['target'] / tmp_success['session_id'], 3)
        fill_value = tmp_success[(tmp_success.session_id < 150) & (
                tmp_success[column + '_success'] < 0.06)][column + '_success'].mean()
        tmp_success.loc[tmp_success.session_id < 30, column + '_success'] = np.nan
        tmp_success.loc[
            (tmp_success.session_id < 150) & (tmp_success[column + '_success'] < 0.06),
            column + '_success'] = np.nan
        tmp_success[column + '_success'].fillna(fill_value, inplace=True)
        tmp_success = tmp_success[['region', column, column + '_success']].drop_duplicates(
            subset=['region', column])
        df = pd.merge(df, tmp_success, how='left', on=['region', column])

    return df


def create_reference(df_train):
    '''
    Takes full scope of original df and applies all transformations
    to create reference data for undersampled training df.
    Returns reference df
    '''

    df_reference = fill_nas(df_train)
    df_reference = add_prev_activity(df_reference)
    df_reference = add_ad_startdates(df_reference)
    df_reference = add_region(df_reference)
    df_reference = modify_device_city(df_reference)
    df_reference = transform_cat_feats(df_reference)
    df_reference = drop(df_reference)

    return df_reference
