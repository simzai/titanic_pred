TITANIC SURVIVAL PREDICTION

OVERVIEW-----------------------------------------------------------------------------
This project builds a machine learning model to predict Titanic passenger survival based on passenger information. The dataset is sourced from the Kaggle Titanic dataset.The model uses XGBoost, a gradient boosting algorithm, to achieve high accuracy by combining engineered features and proper preprocessing.

FEATURE DESCRIPTION-------------------------------------------------------------------

age:	Age of the passenger

fare:	Ticket fare paid

sex_male:	Binary encoding: male=1, female=0

embarked_C:	One-hot encoding for port C

embarked_Q:	One-hot encoding for port Q

embarked_S:	One-hot encoding for port S

pclass:	Passenger class (1=First, 2=Second, 3=Third)

alone:	Boolean indicating if passenger is alone

family_size:	Total family members aboard (sibsp + parch + 1)

deck_encoded:	Encoded deck letter (A–G)

TARGET VARIABLE:
survived (0 = No, 1 = Yes)

Handling missing values

age and embarked were filled using mean or most frequent strategy.

deck was filled using passenger class mapping (C, E, G based on Pclass).

FEATURE ENGINEERING---------------------------------------------------------------------

Combined sibsp + parch → family_size.

Encoding:Categorical variables like sex, embarked → one-hot encoding.

Deck letters → label encoded.

SCALING::Numerical features (age, fare) were scaled for better performance in certain algorithms.

MODEL-----------------------------------------------------------------------------------
Algorithm: XGBoost Classifier

Hyperparameters after tuning:

n_estimators: 400

max_depth: 3

learning_rate: 0.01

subsample: 0.7

colsample_bytree: 0.7

PERFORMANCE------------------------------------------------------------------------------

Cross-validation accuracy: 83.7%

Test accuracy: 81.6%

Recall for survivors: 0.70

Precision for survivors: 0.83

Top features: sex_male, pclass, fare, age, deck_encoded, family_size.
