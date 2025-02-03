README file for Sberavtopodpiska webservice target actions prediction

Author: E.Deripaskina
Date: 22.01.2024

The SCOPE OF THE PROJECT includes:
1) studying and preprocessing client's data on users visiting their website (collected by Google Analytics),
2) selection and training of the most appropriate model for website visitors activity prediction,
3) prediction webservice deployment.

Raw DATA provided by the client:

ga_hits.csv
ga_sessions.csv

ga_sessions file contains attributes of each session, such as visitor's geo_location, utm data, visit number,
visit date etc.
ga_hits file describes previous sessions in detail specifying different actions performed by the visitor,
time of visit, client_id etc.

Attributes in ga_sessions are to be used as predictors. 
Data in ga_hits are to be used to extract target value and some previous activity information per client.
Target value is defined by presence or absence of target actions in the session description in ga_hits.
List of target actions was provided by the client.

PROJECT STRUCTURE:

root:

README.txt
requirements.txt

main.py
  This file processes webservice requests.
  It provides webservice status and version.
  Also it takes predefined number of attributes in json format and returns prediction.
  Prediction function requires reference files to be present in 'data\' folder and
  pickled pipeline present in 'models\' folder.

folder 'data\' contains:

- original files provided by the client: 
 
   (ga_hits.csv, ga_sessions.csv)
 
- publically available geographic data file to be used at feature engineering stage:
 
   (worldcities.csv)
 
- reference files created separately before data preprocessing stage:

   df_reference.csv
      This file contains categorical column values and their probabilities of target action,
      these probabilities are used to replace categorical values at data preprocessing stage;

   df_previous_activity.csv
      This file contains valuable previous sessions information per client and visit_number
      acquired from ga_hits.
      Columns in this file are to be added as new features at data preprocessing stage;

folder 'modules\' contains:

- python file df_reference.py
    This file defines functions necessary to create dataframe for training
    and two reference dataframes;
  

- python file pipeline.py
    This file imports df_reference functions and creates dataframes.
    Then it defines pipeline steps to preprocess data,
    trains several models and chooses the best one based on their ROC-AUC scores.
    The best model is pickled and saved to 'models\' folder;

folder 'models\' contains:

- sberavto_prediction_pipe.pkl
    Pickled pipeline with data preprocessing and prediction steps ready for deployment

folder 'notebook' contains:

- Jupiter Notebook with conducted research, graphs and detailed description of the suggested
  approach and all preprocessing and training steps


GENERAL DESCRIPTION of the dataset:

- highly imbalanced dataset (less than 3% of sessions with target actions)

- very limited valuable attributes available for prediction, no client profile information
  except geolocation

- none of the available data allows for high accuracy prediction of positive target values,
  probability of target action is at most close to 30% or 40% in a very limited number of the most
  successful combinations of attributes (containing very few sessions)

- high number of unique values in multiple columns that could be left out at model fitting

- dataset populated by automatic means so there is practically no risk of human errors in the data

- some sessions are missing from the dataframe (missing visit numbers for client_ids)

GENERAL APPROACH:

Apply undersampling to balance uneven data.
Create clients' previous activity dataset and use available data as predictors.
Transform categorical features into numeric success probabilities.

PROS oF TRANSFORMATION:

the whole dataset can be used for probabilities calculation. So even when most sessions will be dropped
at undersampling stage the information they contained will still have influence over model fitting;

no need to drop or group unique values manually to limit their number before encoding and model fitting;

at training stage: exceptionally low memory usage as compared to OHE-encoding approach and
fast model fitting.

CONS:

as categories are replaced with their probabilities some nuances will definitely be lost
(however, most of these will be lost anyway if some form of undersampling is applied);

final webservice prediction quality will depend on the frequency of both reference files updates,
model will need constant refitting;

slow processing of prediction requests due to the use of external files that need uploading
(time to get reply (less than 1 second) is still way below
maximum acceptable threshold set at 3 seconds).


RESULTS:

The model has ROC-AUC score of about 0.707.
That is the best score I managed to achieve with the available data, and it still leaves much to be
desired.
The final decision function can be adapted to client's requirements to have either less False Positives
or less False Negatives if needed.
Improvement of models accuracy is possible through setting up some system of
registering detailed client's previous activity.
Also, it is not recommended to keep the same encoded names for adcontents that are used over
several marketing campaigns.
The model could be helpful for regional results evaluation and business analysis, but it could hardly be
considered a reliable target prediction source.
Please use it at you own risk and let your common sense be your guide!

More detailed information on the suggested approach and preprocessing steps is available
in Jupiter notebook 'notebook\Sberavto_final_project.ipynb'

 