import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
#from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

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

df=data[select]

# DATA VISUALIZATION
# First examine how many wounded and killed per year
fig, ax = plt.subplots()
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
assert counts.sum()==181691, 'Counts not equal to size of dataset'
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
# known group name incidents, over 18% by decade. Let's see which regions these
# incidents from the top ten are performed in.

def map_decade(year1,year2):
    dec=df.loc[df['iyear'].between(year1,year2)]
    ten=dec.groupby(['gname'])['iyear'].agg('count')
    ten=ten.sort_values(ascending=False)
    top_ten_data=dec.loc[(dec['gname'].isin(ten[1:11].index))]
    loc_data=top_ten_data[['longitude','latitude','gname']].fillna(0)
    loc_data=loc_data.loc[loc_data['latitude']!=0]
    marker_data=list(zip(loc_data['latitude'], loc_data['longitude'],
                       loc_data['gname']))

    locations=list(zip(loc_data['latitude'], loc_data['longitude']))

    popups = ['lat:{}<br>lon:{}<br>group:{}'.format(lat, lon, group) for (lat,
              lon, group) in marker_data]

    map1=folium.Map(location=[np.mean(loc_data['longitude']),
                    np.mean(loc_data['latitude'])], zoom_start=1,
                    tiles='Cartodb Positron')

    marker_cluster=MarkerCluster(locations=locations,
                                 popups=popups).add_to(map1)

    map1.save(os.path.join('results', 'test_map.html'))
    return map1

map_decade(2010,2017) #adjust years to what you need

# PREPROCESSING
# Create function to remove punctuation, reduce whitespace to single space, lowercase, and
# lemmatize text data


def lemmatize_text(text):
    '''Lemmatize text.'''
    wnl=WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return [wnl.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]


def text_preprocess(data):
    '''Preprocesses text data in specified columns in dataframe.'''
    text_df=data.astype('U')
    lem_text=pd.DataFrame()

    for col in text_df.columns:
        print('working... now on %s' %text_df[col].name)
        reg1=re.compile('[,\.!?"($)@:;~/-]')
        reg2=re.compile('\d+')
        reg3=re.compile('\s+')
        for row in range(len(text_df[col])):
            if text_df[col][row]!='NaN':
                text_df[col][row]=reg1.sub(' ', text_df[col][row])
                text_df[col][row]=reg2.sub(' ', text_df[col][row])
                text_df[col][row]=reg3.sub(' ', text_df[col][row]).strip()
                text_df[col][row]=text_df[col][row].lower()

        lem_text[col]=text_df[col].apply(lemmatize_text)

        for row in range(len(lem_text[col])):
            if lem_text[col][row]!=['nan']:
                text_in_row=' '.join(lem_text[col][row])
                lem_text.loc[row, col]=text_in_row
            else:
                lem_text.loc[row,col]=np.nan
    return lem_text

text_cols=df[['summary','motive','addnotes']]
lem_text=text_preprocess(text_cols)


# Let's factorize our group_id to prepare for model creation.
df['group_id'] = df['gname'].factorize()[0]
gname_to_id=df[['gname', 'group_id']].drop_duplicates().sort_values('group_id')
gname_to_id_dict=dict(gname_to_id.values)
id_to_gname_dict=dict(gname_to_id[['gname','group_id']].values)

#see if chi2 gives best correlated words
N = 2
for gname, group_id in sorted(gname_to_id.items()):
  features_chi2 = chi2(features, labels == group_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(gname))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# Let's also separate the known and unknown data into a pseudo-train and test
# set. Test set is unknown gname and pseudo-train set is known gname

known=df.loc[df['gname']!='Unknown']
x_train=known.drop([x for x in known.columns if df[x].dtype=='object'], axis=1)
x_test=known.gname.astype('category')
unknown=df.loc[df['gname']=='Unknown']
y_train=unknown.drop([x for x in unknown.columns if df[x].dtype=='object'],
                     axis=1)

list_99=['nperpcap', 'nhostkid', 'nhours', 'nhostkid', 'nperps',
         'ransomamtus', 'ransompaidus', 'nreleased']

x_train.corr()
sns.heatmap(x_train.corr())
plt.show()

# Now we can do preprocessing

columns=x_train.columns
x_train=x_train.replace([-9,-99], np.nan)
imp=Imputer(missing_values=np.nan,strategy='most_frequent', axis=1)
x_train=imp.fit_transform(x_train)
x_train=pd.DataFrame(x_train)
x_train.columns=columns

columns=y_train.columns
y_train=y_train.replace([-9,-99], np.nan)
imp=Imputer(missing_values=np.nan,strategy='most_frequent', axis=1)
y_train=imp.fit_transform(y_train)
y_train=pd.DataFrame(y_train)
y_train.columns=columns

# Scale for dimensionality reduction and to standardize data.

scale=StandardScaler()
x_train=scale.fit_transform(x_train)
y_train=scale.fit_transform(y_train)

# create holdout

processed_data=pd.concat([pd.DataFrame(x_train).reset_index(drop=True),
                          x_test.reset_index(drop=True)], axis=1,
                            ignore_index=True)
dataset, holdout= train_test_split(pd.DataFrame(processed_data),test_size=0.2)
dataset_lbl=dataset[:][43].reset_index(drop=True)
dataset.reset_index(drop=True, inplace=True)
dataset=dataset.iloc[:,:-1]

# PCA on scaled data
pca=PCA()
pca.fit(x_train)
pca.components_

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
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(dataset, dataset_lbl)
logisticRegr.score(dataset, dataset_lbl)
hold_pred=logisticRegr.predict(holdout.iloc[:,:-1])
accuracy_score(hold_pred, holdout[:][43])

pred=logisticRegr.predict(y_train)
pred=pd.DataFrame(pred)

logisticRegr.score(dataset,dataset_lbl)



