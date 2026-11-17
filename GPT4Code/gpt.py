import openai
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

openai.api_key = "<key>"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

df = pd.read_csv("Dataset/test_annotated.csv")
X = df["code_block"].values

def start_prompting(file_name):

  print("start prompting...")

  results = []

  for x in X:
      
    messages = [
      {
        "role": "system", 
        "content": '''
        You are a specialist in machine learning. 
        Data leakage in machine learning code occurs when information from the test set or the future is used during the training phase. This leads to overly optimistic model performance estimates, as the model learns patterns that won't generalize well to new, unseen data.
        You will be provided with a piece of Python code, and your task is to decide if it contains data leakage or not. 
        Answer with yes or no.'''
        },
      # #1
      #   {
      #   "role": "user", 
      #   "content": '''
      #     real_test_data.datetime = real_test_data.datetime.apply(pd.to_datetime)
      #     real_test_data['month'] = real_test_data.datetime.apply(lambda x : x.month)
      #     real_test_data['hour'] = real_test_data.datetime.apply(lambda x : x.hour)
      #     real_test_data.head()
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "No"
      #   },
      #   {
      #   "role": "user", 
      #   "content": '''
      #     scaler = MinMaxScaler()
      #     selected_columns = ['median_income', 'total_rooms','population','median_age']
      #     SC = scaler.fit_transform(data[selected_columns])
      #     x_train, x_test, y_train, y_test = train_test_split(SC, data['median_house_value'], test_size=0.20)
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "Yes"
      #   },
      # #2
      #   {
      #   "role": "user", 
      #   "content": '''
      #     countries = X_Train.Country.unique()

      #     df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
      #     df_out2 = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

      #     for country in countries:
      #         states = X_Train.loc[X_Train.Country == country, :].State.unique()
      #         #print(country, states)
      #         # check whether string is nan or not
      #         for state in states:
      #             X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]
                  
      #             y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']
      #             y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']
                  
      #             X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]
                  
      #             X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)
      #             X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])
                  
      #             X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]
                  
      #             X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
      #             X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]
                  
      #             X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)
      #             X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])
                  
      #             # XGBoost
      #             model1 = XGBRegressor(n_estimators=2000)
      #             model1.fit(X_Train_CS, y1_Train_CS)
      #             y1_pred = model1.predict(X_Test_CS)
                  
      #             model2 = XGBRegressor(n_estimators=2000)
      #             model2.fit(X_Train_CS, y2_Train_CS)
      #             y2_pred = model2.predict(X_Test_CS)
                  
      #             # LightGBM
      #             model3 = lgb.LGBMRegressor(n_estimators=2000)
      #             model3.fit(X_Train_CS, y1_Train_CS)
      #             y3_pred = model3.predict(X_Test_CS)
                  
      #             model4 = lgb.LGBMRegressor(n_estimators=2000)
      #             model4.fit(X_Train_CS, y2_Train_CS)
      #             y4_pred = model4.predict(X_Test_CS)
                  
      #             df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})
      #             df2 = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y3_pred, 'Fatalities': y4_pred})
      #             df_out = pd.concat([df_out, df], axis=0)
      #             df_out2 = pd.concat([df_out2, df2], axis=0)
      #         # Done for state loop
      #     # Done for country Loop
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "No"
      #   },
      #   {
      #   "role": "user", 
      #   "content": '''
      #     grid = dict()
      #     grid['n_estimators'] = [1000]
      #     grid['max_depth'] = [125,150,175]
      #     cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
      #     model = RandomForestRegressor()

      #     search = GridSearchCV(model,param_grid=grid, scoring='neg_mean_squared_error', cv=cv)
      #     results = search.fit(X, Y)
      #     # summarize
      #     print('MSE (NEGATIVE): %.3f' % results.best_score_)
      #     print('Config: %s' % results.best_params_)
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "Yes"
      #   },
      # #3
      #   {
      #   "role": "user", 
      #   "content": '''
      #     data_merged = data.merge(store, on='Store',how='left')
      #     print(data.shape)
      #     print(data_merged.shape)
      #     print(data_merged.isna().sum().sum()) #to cross check if there are any missing values
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "No"
      #   },
      #   {
      #   "role": "user", 
      #   "content": '''
      #     df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
      #     df['CompetitionDistance'] = np.log(df.CompetitionDistance) + 1
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "Yes"
      #   },

      #   #4
      #   {
      #   "role": "user", 
      #   "content": '''
      #     "#Without Scaling Data

      #     rfr.fit(X_train,y_train)
      #     y_pred=rfr.predict(X_test)
      #     submission2=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
      #     submission2['count']=y_pred
      #     submission2.to_csv('submissionII.csv',index=False)               
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "No"
      #   },
      #   {
      #   "role": "user", 
      #   "content": '''
      #     model = Ridge(alpha=330)
      #     model.fit(X, Y)

      #     test_predicted = pd.DataFrame()
      #     test_predicted['Id'] = test_data.Id
      #     test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))
      #     test_predicted.to_csv('submission-ridge-330.csv', index=False)
      #     test_predicted.describe()
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "Yes"
      #   },

      #   #5
      #   {
      #   "role": "user", 
      #   "content": '''
      #     X_scaler_ft = MinMaxScaler()
      #     X_train_ft = X_scaler_ft.fit_transform(train_X_ft)
      #     X_val_ft =  X_scaler_ft.transform(val_X_ft) # intput/output 2D array-like

      #     y_scaler_ft = MinMaxScaler()
      #     y_train_ft = y_scaler_ft.fit_transform(train_y_ft)
      #     y_val_ft = y_scaler_ft.transform(val_y_ft) # array-like
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "No"
      #   },
      #   {
      #   "role": "user", 
      #   "content": '''
      #     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
      #     features = tfidf.fit_transform(data.Text).toarray()
      #     print(features)
      #     labels = data.category_id
      #     print(labels)
      #     features.shape  
      #     '''
      #   },
      #   {
      #   "role": "assistant", 
      #   "content": "Yes"
      #   },
        
        {
        "role": "user", 
        "content": x
        }
      ]

    completion = completion_with_backoff(
      model="gpt-3.5-turbo-1106", 
      messages=messages,
    )

    print(completion.choices[0].message.content)
    results.append(completion.choices[0].message.content)


  df = pd.DataFrame(results)
  df.to_csv("../Results/GPT/"+file_name+".csv")
  print("Done!")


start_prompting("zero_shot")

###################
# Testing
###################
# df = pd.read_csv("Dataset/test_annotated.csv")
# Y_true = df["has_data_leakage"].values

# df = pd.read_csv("gpt_results_zero_shot.csv")
# Y_pred = df["has_data_leakage"].values

# print(1.0 * np.sum(Y_true == Y_pred) / len(Y_true))
# print(precision_score(Y_true, Y_pred))
# print(recall_score(Y_true, Y_pred))
# print(f1_score(Y_true, Y_pred))
# print(fbeta_score(Y_true, Y_pred, beta=2))
