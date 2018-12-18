import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
#import nltk
#from nltk.stem import WordNetLemmatizer
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# Populating interactive namespace

data=pd.read_excel(os.getcwd()+'\\Dataset\\globalTerrorDB.xlsx')

# Inspect data
print(data.info())
print(data.columns)
print(data.describe())

# Read codebook associated with dataset. After reading codebook and
# inspecting dataset we can begin by dropping data that won't contribute
# to the model fit.

idx_data=data.eventid

# Let's select data that seems most complete and important to naming the
# group responsible according to the codebook and inspection of dataset. 135+
# columns after feature engineering would also be computationally expensive to
# fit a model.

select=['iyear','imonth','iday','country', 'region',
        'latitude', 'longitude', 'summary', 'alternative',
        'multiple', 'success', 'suicide', 'attacktype1',
        'targtype1', 'natlty1',
        'gname','motive','nperps','nperpcap', 'claimed',
        'claimmode', 'weaptype1',
        'weapsubtype1', 'weapdetail', 'nkill', 'nkillus',
        'nkillter', 'nwound', 'nwoundus','nwoundte', 'property','propextent',
        'propvalue', 'ishostkid', 'nhostkid', 'nhostkidus',
        'nhours','ndays', 'divert', 'kidhijcountry','ransom', 'ransomamtus',
        'ransompaidus', 'hostkidoutcome', 'nreleased',
        'addnotes', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY'
        ]

# Note: gname is target variable
# We remove groups that perform less than 3 incidents to not add confounding
# factors to more established groups.

df=data[select]
gname_counts=df['gname'].value_counts()
gname_counts[gname_counts<3]
gname_list=gname_counts[gname_counts >= 3].index.tolist()
df=df[df['gname'].isin(gname_list)]

# DATA VISUALIZATION
# First examine how many wounded and killed per year

plt.figure(figsize=(12,5))
g1=df.groupby(['iyear'])['nwound'].sum().reset_index(name='counts')
idx=g1['iyear']
width = np.min(np.diff(idx))/2
y1=g1['counts']
rects1=plt.bar(idx-width/2,y1,width,label='Wounded')


g2=df.groupby(['iyear'])['nkill'].sum().reset_index(name='counts2')
y2=g2['counts2']
rects2=plt.bar(idx+width/2,y2,width,label='Killed')
plt.xlabel('Year')
plt.ylabel('Counts')
plt.xlim(1970, 2018)
plt.xticks(idx, rotation=45)
plt.legend()
plt.title('Number of Killed and Wounded per Year')
plt.tight_layout()
plt.show()
#plt.savefig('Number of Killed and Wounded per Year.png')

# Missing data for 1993 is explained in the codebook.

# As time progresses closer to the present, we can see that the ratio of
# number of wounded to number killed becomes more balanced instead of skewed
# in the number of wounded direction. This could be due to a number of factors
# including better reporting of fatalities, more fatal weaponry, and an
# increase in the number of terrorist attacks.

# Let's inspect the number of terror incidents per year

g3=df.groupby(['iyear']).agg('count')
counts=g3['imonth']
assert counts.sum()==179016, 'Counts not equal to size of dataset'
plt.figure(figsize=(12,5))
plt.bar(g3.index, counts)
plt.xticks(idx, rotation=45)
plt.title('Number of Terror Incidents per Year')
plt.tight_layout()
plt.show()
#plt.savefig('Number of Terror Incidents per Year.png')

# Number of terror incidents seems to have peaked in 2014 and seems to be
# steadily decreasing. Let's see how correlated the three variables are:
# wounded, killed, and number of incidents with a heatmap of the three.

hmap=pd.DataFrame()
hmap['nwoundyear']=g1['counts']
hmap['nkillyear']=g2['counts2']
hmap['incidents']=g3['imonth'].values
sns.heatmap(hmap.corr())
print(hmap.corr())
plt.show()

# As expected the three are heavily correlated. Perhaps we should inspect
# other variables that may not be as obviously correlated. Let's create a
# general heatmap.

# Before that, let's look at a visualization of the number of incidents
# evoked by group. Sorted by count and displaying top 10 by decade


def incidentGraph(year_beginning, year_ending):
    '''Creates a map of the incidents between specified years. Format:
        incidentGraph(year_beginning, year_ending)'''
    dec=df.loc[df.iyear.between(year_beginning, year_ending)]
    g=dec.groupby(['gname'])['iyear'].agg('count')
    g=g.sort_values(ascending=False)
    plt.figure(figsize=(8,4))

# We have a lot of incidents that are unknown. More unknown incidents than
# known ones. If we look at just the first 10 known actors, then we can see
# a better layout of the known terrorist groups.

    plt.bar(g[1:11].index, g[1:11])
    plt.xticks(rotation=30,horizontalalignment='right')
    plt.title('Terror Groups by Count Ignoring Unknowns between %s and %s'
              % (year_beginning, year_ending))
    plt.tight_layout()
    plt.show()
    per=g[1:11].sum()/g.sum() *100
    print('The top 10 groups perform %s%% of incidents' % per.round(2))

incidentGraph(1970,1979)
incidentGraph(1980,1989)
incidentGraph(1990,1999)
incidentGraph(2000,2009)
incidentGraph(2010,2017)

# We can see the main groups responsible for a large portion of the number of
# known group name incidents, over 18% by decade. Between 2000 and 2009, there
# seems to be a much greater variety of terrorist groups performing attacks.
# This might be because of the rapid rise in news reporting on terrorism in
# the US and other developed countries. We can perform A/B testing on dates
# before and after 9/11/2001 to see if this event triggered a larger number of
# people to incite terror attacks around the world. Let's see which regions
# these incidents from the top ten are performed in.

def map_decade(year1,year2):
    dec=df[['iyear','longitude','latitude','gname']].loc[df.iyear.between(2010,2017)]
    ten=dec.groupby(['gname'])['iyear'].agg('count')
    ten=ten.sort_values(ascending=False)
    top_ten_data=dec.loc[(dec['gname'].isin(ten[1:11].index))]
    top_ten_data['gname']=top_ten_data['gname'].str.replace("'","")
    loc_data=top_ten_data[['longitude','latitude','gname']].fillna(0)
    loc_data=loc_data.loc[loc_data['latitude']!=0]
    marker_data=list(zip(loc_data['latitude'], loc_data['longitude'],
                       loc_data['gname']))

    locations=list(zip(loc_data['latitude'], loc_data['longitude']))

    popups = ['lat:{}<br>lon:{}<br>group:{}'.format(lat, lon, group) for (lat,
              lon, group) in marker_data]

    map1=folium.Map(location=[np.mean(loc_data['longitude']),
                    np.mean(loc_data['latitude'])], zoom_start=2,
                    tiles='Mapbox Bright')
    marker_cluster=MarkerCluster(locations=locations,
                                 popups=popups).add_to(map1)
    map1.save(os.path.join('results', 'test_map.html'))
    return map1

map_decade(2010,2017) #adjust years to what you need

# PREPROCESSING
# The following code snippet has been run and result has been saved to
# save_lemmatized.csv. It took a whole night to run so I've commented it out
# for time's sake.

# Create function to remove punctuation, reduce whitespace to single space,
# lowercase, and lemmatize text data



#def lemmatize_text(text):
#    '''Lemmatize text.'''
#    wnl=WordNetLemmatizer()
#    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
#    return [wnl.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]



#def text_preprocess(data):
#    '''Preprocesses text data in specified columns in dataframe.'''
#    text_df=data.astype('U')
#    lem_text=pd.DataFrame()
#
#    for col in text_df.columns:
#        print('working... now on %s' %text_df[col].name)
#        reg1=re.compile('[,\.!?"($)@:;~/-]')
#        reg2=re.compile('\d+')
#        reg3=re.compile('\s+')
#        for row in range(len(text_df[col])):
#            if text_df[col][row]!='NaN':
#                text_df[col][row]=reg1.sub(' ', text_df[col][row])
#                text_df[col][row]=reg2.sub(' ', text_df[col][row])
#                text_df[col][row]=reg3.sub(' ', text_df[col][row]).strip()
#                text_df[col][row]=text_df[col][row].lower()
#
#        lem_text[col]=text_df[col].apply(lemmatize_text)
#
#        for row in range(len(lem_text[col])):
#            if lem_text[col][row]!=['nan']:
#                text_in_row=' '.join(lem_text[col][row])
#                lem_text.loc[row, col]=text_in_row
#            else:
#                lem_text.loc[row,col]=np.nan
#    return lem_text
#
#text_cols=df[['summary','motive','addnotes']][0:181691]
#lem_text=text_preprocess(text_cols)
#lem_text.to_csv('save_lemmatized.csv')

lem_text=pd.read_csv('save_lemmatized.csv',index_col=0)
lem_text['gname']=df['gname']
lem_text=lem_text[lem_text['gname'].isin(gname_list)]

# Let's factorize our group_id to prepare for model creation. Save dictionaries
df['group_id'] = df['gname'].factorize()[0]
gname_id_df=df[['gname', 'group_id']].drop_duplicates().sort_values('group_id')
gname_to_id_dict=dict(gname_id_df.values)
id_to_gname_dict=dict(gname_id_df[['group_id','gname']].values)

#join text for tfidf

lem_text['joined']=lem_text['summary'].map(str)+' '+lem_text['motive'].map(str)
lem_text['joined']+=' '+lem_text['addnotes'].map(str)

lem_text['joined']=lem_text['joined'].map(lambda x: re.sub('nan', '', x))

#tfidf
tfidf = TfidfVectorizer(stop_words='english', max_features=2000,
                        ngram_range=(1,2), norm='l2', min_df=0.02)
features = tfidf.fit_transform(lem_text.joined.astype('U')).toarray()
labels = df.group_id
features.shape

# Perform chi2 to see top 3 unigrams and bigrams from tfidf vectorizer. Do this
# to get a sense if vectorized words are relevant to groups.

def chi2_test(corpus, feats):

    counter=0
    for gname, group_id in sorted(corpus.items()):
        counter+=1
        if counter == 20:
            break
        else:
            features_chi2 = chi2(feats, labels == group_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams= [v for v in feature_names if len(v.split(' ')) ==2]
            print("# '{}':".format(gname))
            print("  . Most correlated unigrams:\n. {}"
              .format('\n. '.join(unigrams[-3:])))
            print("  . Most correlated bigrams:\n. {}"
              .format('\n. '.join(bigrams[-3:])))

chi2_test(gname_to_id_dict, features)

# We can see that many of the unigrams and bigrams are the same for different
# groups. This is because these groups have no text data to draw from and are
# outputting the "top 3" correlated words in comparison to all documents.

# test which model performs best with text data
def model_testing(features, labels):
    models = [
            RandomForestClassifier(n_estimators=200,
                                   max_depth=3, random_state=42),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(solver='sag', n_jobs=2, tol=0.1,
                          random_state=42),
            ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features[0:10000], labels.iloc[0:10000]
                               , cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx',
                                                   'accuracy'])

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.savefig('results\model_comparison_small_sample.png')
    #plt.show()

model_testing(features, labels)
# looks like linearSVC performs the best on a small sample of the data. Time
# complexity-wise, it is way more computationally expensive, so we will stick
# with logistic regression as our main model.


# Drop object columns as we have vectorized features for them now. Then fix
# issues with negative values as NaN as specified in the codebook. Then
# separate the two dataframes into num_known, text_known, num_unknown,
# text_unknown. Then create pipelines for both numerical features and
# text features and join them in classifier through FeatureUnion.

df=df.replace([-9,-99], np.nan).reset_index(drop=True)

features=pd.DataFrame(features).reset_index(drop=True)

df=pd.concat([df,features],axis=1)


known=df.loc[df['group_id']!=1].reset_index()
unknown=df.loc[df['group_id']==1].reset_index()

known=known.drop([x for x in known.columns if known[x].dtype=='object'],
                 axis=1)
unknown=unknown.drop([x for x in unknown.columns if unknown[x].dtype=='object'],
                     axis=1)

known_group_id=known['group_id']
known.info()
unknown.info()

#imp=Imputer(missing_values=np.nan, strategy='most_frequent', axis=1)
#known.iloc[:,0:44]=imp.fit_transform(known.iloc[:,0:44])
#
#
#scl=StandardScaler()
#known.iloc[:,0:44]=scl.fit_transform(known.iloc[:,0:44])
#known.head()
# Here our unknown data becomes our holdout set of data we want to predict the
# model on. We use train_test_split to train model and test model performance.

x_train, x_test, y_train, y_test= train_test_split(
        known.drop(['group_id'], axis=1),
        known_group_id, test_size=0.2, random_state=42)


# Predict using xgboost as we have nans and zeros in our data. XGBoost "learns"
# how to handle NaN and zeros.

#param={'max_depth': 3,
#       'eta': 0.3,
#       'silent': 1,
#       'n_jobs':3,
#       'objective': 'multi:softmax',
#       'num_class': 1340,
#       'random_state':42,
#       'num_round':30}

#clf=xgb.XGBClassifier(**param)
#clf.fit(x_train,y_train, eval_metric='auc', verbose=True)
#joblib.dump(clf, 'clf_model.pkl', compress=True)

clf=joblib.load('clf_model.pkl')
y_pre=clf.predict(x_test)
print(accuracy_score(y_test, y_pre))
print(classification_report(y_test, y_pre))

real_preds=clf.predict(unknown.drop(['group_id'],axis=1))


#joblib.load('clf_model.pkl') # load it later
# Remove for now. CV is weird.
#
#scores = cross_val_score(logreg, x_train, y_train, cv=3, scoring='accuracy')
#print(scores.mean())
#
#predicted_train=cross_val_predict(logreg, x_train,y_train, cv=3)
#print(accuracy_score(y_train, predicted_train))
#print(classification_report(y_train, predicted_train))
#
#
#predicted_test=cross_val_predict(logreg, x_test, cv=3)
#print(accuracy_score(y_test, predicted_test))
#print(classification_report(y_test, predicted_test))










# Scale for dimensionality reduction and to standardize data.

#scale=StandardScaler()
#x_train=scale.fit_transform(x_train)
#y_train=scale.fit_transform(y_train)

# holdout
#processed_data=pd.concat([pd.DataFrame(x_train).reset_index(drop=True),
#                          x_test.reset_index(drop=True)], axis=1,
#                            ignore_index=True)
#dataset, holdout= train_test_split(pd.DataFrame(processed_data),test_size=0.2)
#dataset_lbl=dataset[:][43].reset_index(drop=True)
#dataset.reset_index(drop=True, inplace=True)
#dataset=dataset.iloc[:,:-1]



# PCA on scaled data
#pca=PCA()
#pca.fit(x_train)
#pca.components_

## Graph PCA features vs. Explained variance
#features=range(pca.n_components_)
#
#plt.bar(features, pca.explained_variance_)
#plt.xticks(features)
#plt.xlabel('variance')
#plt.ylabel('PCA Feature')
#
#plt.show()
#
## Transform data and graph to see if difference in groups
#transformed=pca.transform(x_train)
#print(transformed.shape)

#logreg first
#logisticRegr = LogisticRegression(solver = 'lbfgs')
#logisticRegr.fit(dataset, dataset_lbl)
#logisticRegr.score(dataset, dataset_lbl)
#hold_pred=logisticRegr.predict(holdout.iloc[:,:-1])
#accuracy_score(hold_pred, holdout[:][43])
#
#pred=logisticRegr.predict(y_train)
#pred=pd.DataFrame(pred)
#
#logisticRegr.score(dataset,dataset_lbl)



