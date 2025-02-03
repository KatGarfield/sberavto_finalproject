'''The purpose of this module is to create a single pipeline that will
include all data preprocessing steps and the most accurate model fitted with
the available data.

All custom preprocessing functions will be included in one external function:
preprocess_dataset()
StandardScaler will be added in a separate pipeline step and will be applied
to all features.

After all the necessary steps are defined the imported functions from df_reference.py
will be used to create dataframe with target values for training.
This dataframe will be immediately split into 5 folds.
The training folds will be used to create previous activity and reference dfs
and will be undersampled to balance uneven data.
Then both training and test folds will be fed into the pipeline.
The best model will be chosen automatically based on mean ROC-AUC score and
saved to 'models\' folder.
Before saving, the best model will be fitted with all five folds.
Also, all five folds will be used to update df_previous_activity.csv and
df_reference.csv.
'''

import pandas as pd
import dill

from datetime import datetime
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from modules.df_reference import add_target_column, create_reference, isoutlier_3sigma
import warnings
warnings.filterwarnings('ignore')


def preprocess_dataset(df):

    import numpy as np
    import pandas as pd

    def replace_nas_training(df):
        '''
        Fills NAs in device_brand column with 'Apple' if device browser is Safari
        Fills NAs in columns with numeric datatype with median value
        Fills NAs in columns with object datatype with '(not set)'
        Merge visit date and time into visit datetime object
        '''

        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numerical_features:
            df[column] = df[column].fillna(df[column].median())

        df.visit_date = df.visit_date.fillna(df.visit_date.mode())
        df.visit_date = df.visit_date.astype('datetime64[ns]')

        categorical_features = df.select_dtypes(include=['object']).columns
        for column in categorical_features:
            df[column] = df[column].fillna('(not set)')

        return df

    def add_client_history(df):
        '''
        Merges original dataframe with previous activity dataframe created earlier
        on client_id and visit_number columns,
        then offsets visit number 5 times to fill some of the NA values for visits
        skipped in the original df (and for that reason also missing in the previous activity df)

        Calculates days since client's previous visit as a difference between current visit_date
        and visit_date from previous activity df.

        Fills all NAs for first time clients with no history and for returning clients without
        previous data with median values.

        Returns modified original df
        '''

        # Copy df for multiple merge

        df = pd.merge(df, df_prev, on=['client_id', 'visit_number'], how='left')
        df['days_after_last_visit'] = ((df['visit_date'] - df['prev_visit']) /
                                       np.timedelta64(1, 'D'))

        # Fill missing values and first time client values in columns (mean_result,
        # mean_action, prev_session_duration) with 0

        new_cols = ['mean_result', 'mean_actions', 'prev_session_duration']
        for col in new_cols:
            df[col] = df[col].fillna(0)

        # Change outliers to high threshold

        low_threshold, high_threshold = isoutlier_3sigma(df.days_after_last_visit)
        df['days_after_last_visit'] = df['days_after_last_visit'].apply(
            lambda x: x if x < high_threshold else high_threshold)
        df['days_after_last_visit'].fillna(
            high_threshold, inplace=True)

        # Change outliers in visit number column to high threshold value

        low_threshold, high_threshold = isoutlier_3sigma(df.visit_number)
        df['visit_number'] = df['visit_number'].apply(
            lambda x: x if x < high_threshold else high_threshold)

        return df

    def add_time_features(df):
        '''
        Creates time related new features

        a) Calculates how long ago the ad was launched
           (number of days is defined as a difference between current session date
            and the minimum date found in the reference dataframe)
            and replaces outliers with high threshold value
        b) Adds new date column based on visit_date and transforms it into numeric as follows:
            year multiplied by 365 days + month multiplied by 30.5 days + day number
        c) Adds day of month column

        Returns modified df
        '''

        df_ad = df_reference[['utm_adcontent', 'ad_start_tmp']].drop_duplicates(subset=['utm_adcontent'])
        df = pd.merge(df, df_ad, how='left', on='utm_adcontent')
        df['days_since_ad_st'] = round((df.visit_date - df.ad_start_tmp) /
                                       np.timedelta64(1, 'D'))
        df['days_since_ad_st'] = df['days_since_ad_st'].fillna(
                df['days_since_ad_st'].median())

        # Change outliers to high_threshold

        low_threshold, high_threshold = isoutlier_3sigma(df.days_since_ad_st)
        df['days_since_ad_st'] = df['days_since_ad_st'].apply(
            lambda x: x if x < high_threshold else high_threshold)

        # Bring date and time columns to numeric type

        df['visit_dt'] = df['visit_date'].dt.date.apply(lambda x: (
                int(str(x).split("-")[0]) * 365 +
                int(str(x).split("-")[1]) * 30 +
                int(str(x).split("-")[2])))

        return df

    def add_geo_features(df):
        '''
        Applies region and distance_to_capital data from reference
        Fills NAs with 'Russia' and distance standard deviation
        Replaces outliers with high threshold
        '''

        region_to_merge = df_reference[['geo_city', 'region']].drop_duplicates(subset=['geo_city'])
        df = pd.merge(df, region_to_merge, how='left', on='geo_city')
        df['region'] = df['region'].fillna('Russia')

        return df

    def prepare_device_medium_campaign(df):
        '''
        Joins device screen resolution and device brand into one column that supposedly
        acts as a replacement for a more valuable but mostly empty device model column

        Joins different forms of paid ads (cpc, cpm etc.) into bigger groups
        Replace values with small number of sessions with 'other medium'

        Adds major campaign names from reference to other utm columns

        Return modified df
        '''

        df.device_screen_resolution = (df.device_screen_resolution +
                                       df.device_brand)

        major_campaigns = list(df_reference.major_campaign.unique())
        dependency_cols = ['utm_source', 'utm_keyword', 'utm_adcontent']
        df['major_campaign'] = df.utm_campaign.apply(
            lambda x: x[:5] if x in major_campaigns else '')
        for column in dependency_cols:
            df[column] = df['major_campaign'] + df[column]

        return df

    def apply_coefficients(df):
        '''
        Add success probabilities from reference data to all categorical columns
        Returns modified df
        '''

        city_to_merge = df_reference[['geo_city', 'geo_city_success']].drop_duplicates(
            subset=['geo_city'])
        df = pd.merge(df, city_to_merge, how='left', on=['geo_city'])
        df['geo_city_success'] = df['geo_city_success'].fillna(city_to_merge.geo_city_success.median())

        device_to_merge = df_reference[[
            'device_screen_resolution', 'device_screen_resolution_success']].drop_duplicates(
            subset=['device_screen_resolution'])
        df = pd.merge(df, device_to_merge, how='left', on=['device_screen_resolution'])
        df['device_screen_resolution_success'] = df['device_screen_resolution_success'].fillna(
            device_to_merge.device_screen_resolution_success.median())

        campaign_to_merge = df_reference[[
            'region', 'utm_campaign', 'utm_campaign_success']].drop_duplicates(
            subset=['region', 'utm_campaign'])
        df = pd.merge(df, campaign_to_merge, how='left', on=['region', 'utm_campaign'])
        df['utm_campaign_success'] = df['utm_campaign_success'].fillna(
            campaign_to_merge.utm_campaign_success.median())

        source_to_merge = df_reference[[
            'region', 'utm_source', 'utm_source_success']].drop_duplicates(
            subset=['region', 'utm_source'])
        df = pd.merge(df, source_to_merge, how='left', on=['region', 'utm_source'])
        df['utm_source_success'] = df['utm_source_success'].fillna(
            source_to_merge.utm_source_success.median())

        adcontent_to_merge = df_reference[[
            'region', 'utm_adcontent', 'utm_adcontent_success']].drop_duplicates(
            subset=['region', 'utm_adcontent'])
        df = pd.merge(df, adcontent_to_merge, how='left', on=['region', 'utm_adcontent'])
        df['utm_adcontent_success'] = df['utm_adcontent_success'].fillna(
            adcontent_to_merge.utm_adcontent_success.median())

        keyword_to_merge = df_reference[[
            'region', 'utm_keyword', 'utm_keyword_success']].drop_duplicates(
            subset=['region', 'utm_keyword'])
        df = pd.merge(df, keyword_to_merge, how='left', on=['region', 'utm_keyword'])
        df['utm_keyword_success'] = df['utm_keyword_success'].fillna(
            keyword_to_merge.utm_keyword_success.median())

        return df

    def drop_columns(df):
        '''
        Drops useless columns before training
        '''

        cols_to_drop = [
            'session_id',
            'client_id',
            'device_brand',
            'device_category',
            'device_browser',
            'device_model',
            'device_os',
            'region',
            'geo_country',
            'utm_medium',
            'prev_visit',
            'visit_date',
            'visit_time',
            'ad_start_tmp',
            'geo_city',
            'device_screen_resolution',
            'utm_campaign',
            'utm_source',
            'utm_adcontent',
            'utm_keyword',
            'utm_medium',
            'major_campaign',
        ]
        df.drop(columns=cols_to_drop, inplace=True)

        return df

    df = replace_nas_training(df)
    df = add_client_history(df)
    df = add_time_features(df)
    df = add_geo_features(df)
    df = prepare_device_medium_campaign(df)
    df = apply_coefficients(df)
    df = drop_columns(df)

    return df


final_numerical_feats = ['visit_number', 'visit_dt',
                         'mean_actions', 'mean_result',
                         'days_after_last_visit',
                         'device_screen_resolution_success',
                         'geo_city_success', 'utm_keyword_success',
                         'prev_session_duration', 'days_since_ad_st',
                         'utm_source_success', 'utm_campaign_success',
                         'utm_adcontent_success']

# Define preprocessor and model pipeline

preprocessor = Pipeline(steps=[
    ('preprocess dataset', FunctionTransformer(preprocess_dataset)),
    ('scaler', ColumnTransformer(transformers=[
        ('standard_scaler', StandardScaler(), final_numerical_feats)])),
    ])

# List suggested models with hyperparameters (hyperparameters selection
# can be found in 'notebook\').

models = (
    RandomForestClassifier(bootstrap=True, max_depth=30,
                           min_samples_leaf=15, min_samples_split=40,
                           n_estimators=100, random_state=26,
                           class_weight='balanced'),
    LGBMClassifier(boosting_type='dart', learning_rate=0.1,
                   max_depth=20, num_leaves=50, n_estimators=500,
                   random_state=26, class_weight='balanced'),
    LogisticRegression(C=0.0001, dual=False, penalty=None,
                       solver='lbfgs', max_iter=1000, random_state=26,
                       class_weight='balanced'),
    GradientBoostingClassifier(max_depth=10, min_samples_leaf=15,
                               min_samples_split=15, n_estimators=50,
                               random_state=26),
    MLPClassifier(activation='logistic', hidden_layer_sizes=(100,),
                  learning_rate_init=0.001, solver='adam', random_state=26),
)

# Create dataset for training

print('Creating training df...')
df_sberavto = add_target_column()

# Custom crossvalidation for undersamped data

best_score = -1
best_precision = -1
model_scores = {}
for model in models:
    model_name = type(model).__name__
    print(model_name)
    model_scores[model_name + '_auc'] = []
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
        ])
    kf = KFold(shuffle=True, random_state=26)
    kf.get_n_splits(df_sberavto)

    lap = 1
    for train_index, test_index in kf.split(df_sberavto):
        print(f'Fold {lap}')
        df_train = df_sberavto.iloc[train_index, :]
        df_test = df_sberavto.iloc[test_index, :]

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        print('Test size:', len(df_test))
        print('Train size:', len(df_train))

        # Create reference and previous activity dfs

        print('Creating reference df...')
        df_reference = create_reference(df_train)
        df_reference.to_csv(r'..\data\df_reference.csv', index=False)
        df_prev = pd.read_csv(r'..\data\df_previous_activity.csv',
                              dtype={'client_id': 'object'},
                              parse_dates=['prev_visit'])

        # Drop extra columns and create df for training

        x_train = df_train.drop(columns=['session_duration', 'result', 'event_action',
                                         'target'])
        y_train = df_train.target

        # Apply undersampling

        rus = RandomUnderSampler(sampling_strategy=0.8, random_state=26)
        x, y = rus.fit_resample(x_train, y_train)

        print(f'Start training {lap}...')
        pipe.fit(x, y)

        # Prepare validation data

        df_test.drop(columns=['session_duration', 'result', 'event_action'], inplace=True)
        x_val = df_test.drop('target', axis=1)
        y_val = df_test.target
        y_pred = pipe.predict(x_val)
        y_pred_proba = pipe.predict_proba(x_val)
        y_pred_proba_train = pipe.predict_proba(x)
        print(f'Training {lap} completed')

        model_scores[model_name + '_auc'].append(roc_auc_score(y_val, y_pred_proba[:, 1]))
        print('Test ROC-AUC:', roc_auc_score(y_val, y_pred_proba[:, 1]))
        print('Train ROC-AUC:', roc_auc_score(y, y_pred_proba_train[:, 1]))
        lap += 1
    pipe_score = sum(model_scores[model_name + '_auc']) / len(model_scores[model_name + '_auc'])
    print('Model:', model_name)
    print('Mean ROC-AUC score:', pipe_score)
    print('5 fold ROC-AUC scores:', model_scores[model_name + '_auc'])
    print('Confusion_matrix:')
    print(confusion_matrix(y_val, pipe.predict(x_val)))
    if pipe_score > best_score:
        best_score = pipe_score
        best_pipe = pipe

print(f'''
best model: {type(best_pipe.named_steps["classifier"]).__name__}
roc_auc: {best_score:.3f}
''')

# Recreate df_reference.csv and df_previous_activity.csv
# Fit and save best model

df_reference_full = create_reference(df_sberavto)
df_reference_full.to_csv(r'..\data\df_reference.csv', index=False)
x_full = df_sberavto.drop(columns=['session_duration', 'result', 'event_action', 'target'])
y_full = df_sberavto.target
rus = RandomUnderSampler(sampling_strategy=0.8, random_state=26)
x, y = rus.fit_resample(x_full, y_full)
best_pipe.fit(x, y)

file_name = r'..\models\sberavto_prediction_pipe.pkl'
with open(file_name, 'wb') as file:
    dill.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'Sberavto Prediction Pipeline',
            'author': 'Ekaterina Deripaskina',
            'version': 1.0,
            'date': str(datetime.now())[:-7],
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'ROC-AUC': best_score,
        }
    }, file, recurse=True)
