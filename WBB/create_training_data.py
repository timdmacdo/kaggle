
# coding: utf-8

# # Preprocess data for machine learning
# 
# ### March 3, 2018
# ### Tiffany Huang
#%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "WDataFiles/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "WDataFiles"]).decode("utf8"))
# In[2]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "WDataFiles/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "WStage2DataFiles"]).decode("utf8"))


# # Load the Training Data

# In[3]:

data_dir = 'WStage2DataFiles/'
df_seeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
df_tourney = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')
df_reg_season = pd.read_csv(data_dir + 'WRegularSeasonCompactResults.csv')
df_field_goal = pd.read_csv(data_dir + 'WFieldGoalPercentage_IDs.csv')
df_rebounds = pd.read_csv(data_dir + 'WReboundMargin_IDs.csv')
df_scoring_offense = pd.read_csv(data_dir + 'WScoringOffense_IDs.csv')
df_turnovers = pd.read_csv(data_dir + 'WTurnovers_IDs.csv')


# # Seeds Data

# In[4]:

df_seeds.head()


# In[5]:

# First, we'll simplify the datasets to remove the columns we won't be using 
# and convert the seedings to the needed format (stripping the regional abbreviation in front of the seed).

# Get just the digits from the seeding. Return as int
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

df_seeds.drop(df_seeds[df_seeds.Season < 2002].index, inplace=True)
df_seeds.index = range(len(df_seeds))
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()


# # Tournament Data

# In[6]:

df_tourney.drop(labels=['DayNum', 'WScore', 'LScore', 'NumOT', 'WLoc'], inplace=True, axis=1)
df_tourney.drop(df_tourney[df_tourney.Season < 2002].index, inplace=True)
df_tourney.index = range(len(df_tourney))
df_tourney.head()


# In[7]:

# Merge the Seeds with their corresponding TeamIDs in the compact results dataframe.
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tourney, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat.drop(labels=['WSeed', 'LSeed'], inplace=True, axis=1)
df_concat.head()


# # Regular Season Data
df_reg_season.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_reg_season.drop(df_reg_season[df_reg_season.Season < 2002].index, inplace=True)
df_reg_season.index = range(len(df_reg_season))
df_reg_season.head()# Find the record during the regular season between two teams
def regSeasonToRecord(df):
    record_dict = {}
    for row in df.itertuples():
        if (row[1] not in record_dict):
            record_dict[row[1]] = {}
        else:
            if ((row[2], row[3]) in record_dict[row[1]]):
                record_dict[row[1]][(row[2], row[3])] += 1
            else:
                record_dict[row[1]][(row[2], row[3])] = 1
    return record_dict

record_dict = regSeasonToRecord(df_reg_season)# Add the record between the winning and losing teams to the table
# Iterate through df_concat rows, look for record in dictionary between WTeam and LTeam and add it to the row
df_record = pd.DataFrame()
for row in df_concat.itertuples():
    if ((row[2], row[3]) in record_dict[row[1]]):
        df_record = df_record.append(pd.DataFrame([record_dict[row[1]][(row[2], row[3])]], columns=['Record']), ignore_index=True)
    elif ((row[3], row[2]) in record_dict[row[1]]):
        df_record = df_record.append(pd.DataFrame([-1.0 * record_dict[row[1]][(row[3], row[2])]], columns=['Record']), ignore_index=True)
    else:
        df_record = df_record.append(pd.DataFrame([0], columns=['Record']), ignore_index=True)

df_concat = pd.concat([df_concat, df_record], axis=1)
df_concat
# # Field Goal Percentage Data

# In[8]:

df_field_goal.drop(labels=['Team', 'Kaggle Name', 'Ranking'], inplace=True, axis=1)
df_field_goal['FG Percentage'] = df_field_goal['FG Percentage'] / 100
df_field_goal['Kaggle TeamID'] = df_field_goal['Kaggle TeamID'].astype(int)
df_field_goal.head()


# In[9]:

df_win_fg = df_field_goal.rename(columns={'Kaggle TeamID':'WTeamID', 'FG Percentage':'WFGP'})
df_lose_fg = df_field_goal.rename(columns={'Kaggle TeamID':'LTeamID', 'FG Percentage':'LFGP'})
df_concat = pd.merge(left=df_concat, right=df_win_fg, how='inner', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lose_fg, how='inner', on=['Season', 'LTeamID'])
df_concat['FG%Diff'] = df_concat.WFGP - df_concat.LFGP
df_concat.drop(labels=['WFGP', 'LFGP'], inplace=True, axis=1)


# # Rebound Margin Data

# In[10]:

df_rebounds.drop(labels=['Team', 'Kaggle Name', 'Ranking', 'Rebound Margin'], inplace=True, axis=1)
df_rebounds['Kaggle TeamID'] = df_rebounds['Kaggle TeamID'].astype(int)
df_rebounds.head()


# In[11]:

df_win_rpg = df_rebounds.rename(columns={'Kaggle TeamID':'WTeamID', 'RPG':'WRPG'})
df_lose_rpg = df_rebounds.rename(columns={'Kaggle TeamID':'LTeamID', 'RPG':'LRPG'})
df_concat = pd.merge(left=df_concat, right=df_win_rpg, how='inner', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lose_rpg, how='inner', on=['Season', 'LTeamID'])
df_concat['ReboundDiff'] = df_concat.WRPG - df_concat.LRPG
df_concat.drop(labels=['WRPG', 'LRPG'], inplace=True, axis=1)
df_concat.head()


# # Scoring Data

# In[12]:

df_scoring_offense.drop(labels=['Team', 'Kaggle Name', 'Rank','Losses','Total Points'], inplace=True, axis=1)
df_scoring_offense['Win%'] = df_scoring_offense['Wins'] / df_scoring_offense['Games Played']
df_scoring_offense['Kaggle TeamID'] = df_scoring_offense['Kaggle TeamID'].astype(int)
df_scoring_offense.head()


# In[13]:

df_win_ppg = df_scoring_offense.rename(columns={'Kaggle TeamID':'WTeamID', 'PPG':'WPPG'})
df_win_ppg.drop(labels=['Games Played','Wins','Win%'], inplace=True, axis=1)
df_lose_ppg = df_scoring_offense.rename(columns={'Kaggle TeamID':'LTeamID', 'PPG':'LPPG'})
df_lose_ppg.drop(labels=['Games Played','Wins','Win%'], inplace=True, axis=1)
df_concat = pd.merge(left=df_concat, right=df_win_ppg, how='inner', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lose_ppg, how='inner', on=['Season', 'LTeamID'])
df_concat['PPGDiff'] = df_concat.WPPG - df_concat.LPPG
df_concat.drop(labels=['WPPG', 'LPPG'], inplace=True, axis=1)
df_concat.head()


# In[14]:

df_win_win_percent = df_scoring_offense.rename(columns={'Kaggle TeamID':'WTeamID', 'Win%':'WWin%'})
df_win_win_percent.drop(labels=['Games Played', 'Wins', 'PPG'], inplace=True, axis=1)
df_lose_win_percent = df_scoring_offense.rename(columns={'Kaggle TeamID':'LTeamID', 'Win%':'LWin%'})
df_lose_win_percent.drop(labels=['Games Played', 'Wins', 'PPG'], inplace=True, axis=1)
df_concat = pd.merge(left=df_concat, right=df_win_win_percent, how='inner', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lose_win_percent, how='inner', on=['Season', 'LTeamID'])
df_concat['Win%Diff'] = df_concat['WWin%'] - df_concat['LWin%']
df_concat.drop(labels=['WWin%', 'LWin%'], inplace=True, axis=1)
df_concat.head()


# # Turnover Data

# In[15]:

df_turnovers.drop(labels=['Team', 'Kaggle Name', 'Ranking'], inplace=True, axis=1)
df_turnovers['Kaggle TeamID'] = df_turnovers['Kaggle TeamID'].astype(int)
df_turnovers.head()


# In[16]:

df_win_turnovers = df_turnovers.rename(columns={'Kaggle TeamID':'WTeamID', 'TOPG':'WTOPG'})
df_lose_turnovers = df_turnovers.rename(columns={'Kaggle TeamID':'LTeamID', 'TOPG':'LTOPG'})
df_concat = pd.merge(left=df_concat, right=df_win_turnovers, how='inner', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lose_turnovers, how='inner', on=['Season', 'LTeamID'])
df_concat['TOPGDiff'] = df_concat.WTOPG - df_concat.LTOPG
df_concat.drop(labels=['WTOPG', 'LTOPG'], inplace=True, axis=1)
df_concat


# In[17]:

# Now we'll create a dataframe that summarizes wins & losses along with 
# their corresponding seed differences. This is the meat of what we'll be creating our model on.
df_wins = pd.DataFrame()
df_wins = df_concat.drop(labels=['Season', 'WTeamID', 'LTeamID'], axis=1)
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['FG%Diff'] = -df_concat['FG%Diff']
df_losses['ReboundDiff'] = -df_concat['ReboundDiff']
df_losses['PPGDiff'] = -df_concat['PPGDiff']
df_losses['Win%Diff'] = -df_concat['Win%Diff']
df_losses['TOPGDiff'] = -df_concat['TOPGDiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions = (df_predictions - df_predictions.min()) / (df_predictions.max() - df_predictions.min())
df_predictions.to_csv('training_data.csv', index=False)


# In[18]:

X_train = df_predictions[['SeedDiff','FG%Diff','ReboundDiff','PPGDiff','Win%Diff','TOPGDiff']].values.reshape(-1,6)
y_train = df_predictions.Result.values

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_train, y_train = shuffle(X_train_normalized, y_train)


# # Train the Model

# In[384]:

## Use a basic logistic regression to train the model. You can set different C values to see how performance changes.
#logreg = LogisticRegression()
#params = {'C': np.logspace(start=-5, stop=3, num=9)}
#clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
#clf.fit(X_train, y_train)
#print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))


## In[ ]:


## In[385]:

#forest = RandomForestClassifier()
#params = {'max_depth': np.logspace(start=2, stop=10, num=13)}
#clf = GridSearchCV(forest, params, scoring='neg_log_loss', refit=True)
#clf.fit(X_train, y_train)
#print('Best log_loss: {:.4}, with best depth: {}'.format(clf.best_score_, clf.best_params_['max_depth']))


#boosting = GradientBoostingClassifier(n_estimators=100)
#params = {'learning_rate': np.logspace(start=0.1, stop=10, num=10)}
#clf = GridSearchCV(boosting, params, scoring='neg_log_loss', refit=True)
#clf.fit(X_train, y_train)

#print('Best log_loss: {:.4}, with best learning rate: {}'.format(clf.best_score_, clf.best_params_['learning_rate']))

n_est = 1000
clf = OneVsRestClassifier(BaggingClassifier(SVC(probability=True),max_samples=1./n_est,n_estimators=n_est))
#params = {'C': np.logspace(start=0.1, stop=10, num=10)}
#clf = GridSearchCV(svc, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)

#print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['learning_rate']))

# In[386]:

df_sample_sub = pd.read_csv(data_dir + 'WSampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


# In[387]:

X_test = np.zeros(shape=(n_test_games, 6))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    #print(year)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed
    
    t1_fgp = df_field_goal[(df_field_goal['Kaggle TeamID'] == t1) & (df_field_goal.Season == year)]['FG Percentage'].values[0]
    t2_fgp = df_field_goal[(df_field_goal['Kaggle TeamID'] == t2) & (df_field_goal.Season == year)]['FG Percentage'].values[0]
    diff_fgp = t1_fgp - t2_fgp
    X_test[ii, 1] = diff_fgp
    
    t1_rpg = df_rebounds[(df_rebounds['Kaggle TeamID'] == t1) & (df_rebounds.Season == year)]['RPG'].values[0]
    t2_rpg = df_rebounds[(df_rebounds['Kaggle TeamID'] == t2) & (df_rebounds.Season == year)]['RPG'].values[0]
    diff_rpg = t1_rpg - t2_rpg
    X_test[ii, 2] = diff_rpg
    
    t1_ppg = df_scoring_offense[(df_scoring_offense['Kaggle TeamID'] == t1) & 
                                (df_scoring_offense.Season == year)]['PPG'].values[0]
    t2_ppg = df_scoring_offense[(df_scoring_offense['Kaggle TeamID'] == t2) & 
                                (df_scoring_offense.Season == year)]['PPG'].values[0]
    diff_ppg = t1_ppg - t2_ppg
    X_test[ii, 3] = diff_ppg
    
    t1_win_percent = df_scoring_offense[(df_scoring_offense['Kaggle TeamID'] == t1) & (df_scoring_offense.Season == year)]['Win%'].values[0]
    t2_win_percent = df_scoring_offense[(df_scoring_offense['Kaggle TeamID'] == t2) & (df_scoring_offense.Season == year)]['Win%'].values[0]
    diff_win_percent = t1_win_percent - t2_win_percent
    X_test[ii, 4] = diff_win_percent
    
    t1_topg = df_turnovers[(df_turnovers['Kaggle TeamID'] == t1) & (df_turnovers.Season == year)]['TOPG'].values[0]
    t2_topg = df_turnovers[(df_turnovers['Kaggle TeamID'] == t2) & (df_turnovers.Season == year)]['TOPG'].values[0]
    diff_topg = t1_topg - t2_topg
    X_test[ii, 5] = diff_topg

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test_normalized = scaler.transform(X_test)


# In[359]:

preds = clf.predict_proba(X_test_normalized)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub


# In[360]:

# Create submission file!
df_sample_sub.to_csv('svc.csv', index=False)


# In[ ]:



