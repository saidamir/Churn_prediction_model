# Python Notebook - Case_Миро

datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns
%matplotlib inline

datasets[0].describe()

datasets[0]['account_id'].nunique()

df_dummies = pd.get_dummies(pd.DataFrame(datasets[0]['integrations_names'].map(lambda x: x.split(', ') if type(x) == str else []).to_list()).stack().reset_index(level=1, drop=True)).groupby(level=0).sum()

#creating new df with dummy columns
df_dum = datasets[0].copy()
df_dum[df_dummies.columns] = df_dummies
df_dum=df_dum.drop(columns='integrations_names')
df_dum.columns

sns.countplot(datasets[0]['paid_status'])


# percent distribution of Paying vs Churned
np.array(datasets[0]['paid_status']=='Churned').sum()/len(datasets[0]['paid_status'])

datasets[0]['integrations_names'].nunique()

# creating Paying and Churned datasets
df1=df_dum.copy() #datasets[0]
df_paying = df1.drop(columns='paid_status')[df1['paid_status']=='Paying']
df_churned = df1.drop(columns='paid_status')[df1['paid_status']!='Paying']
#datasets[0][['account_age',	'seats',	'boards_last_30_days',	
#'projects_last_30_days',	'dau_avg_last_30_days',	'integrations_count',
#'templates_last_30_days']]
pass

# All customers
#df1[['account_age',	'seats',	'boards_last_30_days',	'projects_last_30_days',	'dau_avg_last_30_days',	'integrations_count','templates_last_30_days']]
df1.drop(columns='paid_status').hist(bins = 10, figsize = (10,10))
plt.tight_layout() # descriptions not overlap
pass

# Paying customers
df_paying.hist(bins = 20, figsize = (10,10))
plt.tight_layout() # descriptions not overlap)
plt.title('Paying')
plt.show()
pass



g = sns.PairGrid(datasets[0][['account_age',	'seats',	'boards_last_30_days',	
'projects_last_30_days',	'dau_avg_last_30_days',	'integrations_count',
'templates_last_30_days']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);

age_b = sns.boxplot(data = datasets[0], x = 'paid_status',y='account_age')
age_b.set_xticklabels(rotation = 45, labels = datasets[0]['paid_status'])
pass

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=20)

plt.hist(df_paying['account_age'], **kwargs)
plt.hist(df_churned['account_age'], **kwargs)
pass


df_paying['DAU_divide_seats'] = df_paying['dau_avg_last_30_days']/df_paying['seats']
df_churned['DAU_divide_seats'] = df_churned['dau_avg_last_30_days']/df_churned['seats']


print ('Number of churned seats: ', df_churned['seats'].sum())
print ('Number of paying seats: ', df_paying['seats'].sum())
print ('Mean of churned seats: ', df_churned['seats'].mean())
print ('Number of paying seats: ', df_paying['seats'].mean())
print ('Count of churned seats: ', df_churned['seats'].count())
print ('Count of paying seats: ', df_paying['seats'].count())
print ('Count of churned accounts: ', df_churned['account_id'].count())
print ('Count of paying accounts: ', df_paying['account_id'].count())


print ('No of churned accounts integrations: ', df_churned['integrations_count'].sum())
print ('No of paying accounts integrations: ', df_paying['integrations_count'].sum())

#y axis in where integral of each function equals 1
for column in df1.drop(columns=['paid_status','account_id']):
  
  plt.hist([df_paying[column],df_churned[column]],color=['green', 'red'], density=True )
  plt.title(column)
  plt.legend(['Paying','Churned'])
  plt.show()
  pass
  
  

corr = datasets[0].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

df1=df_dum.copy() #datasets[0]
df2 = df1.set_index('account_id')
y = df2['paid_status']=='Paying' #immediately convert to true and false
x = df2.drop(columns='paid_status')

#find number of nulls
pd.isnull(x).sum()

# replacing NA values with 0
x.fillna(0, inplace=True)

#standardizing
x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y)

clf = LogisticRegression().fit(x_train, y_train)

# don't forget to select the second column!!! as it is probability
y_pred_train, y_pred_test = clf.predict_proba(x_train)[:,1],clf.predict_proba(x_test)[:,1]

print ('ROC_AUC_train: ', roc_auc_score (y_train, y_pred_train), 'ROC_AUC_test: ', roc_auc_score(y_test, y_pred_test))

# fitting the model to the whole data set to determine the most important coefficients
clf = LogisticRegression().fit(x, y)

#coefficients
clf.coef_

#important features in regression, creating a DF for this
#list(zip(df2.drop(columns='paid_status').columns,clf.coef_[0]))
pd.DataFrame({'feature':df2.drop(columns='paid_status').columns,'coef':clf.coef_[0]}).sort_values(by='coef')

#repeating code from the above, as notebook turns off and df are erased from the memory
df_dummies = pd.get_dummies(pd.DataFrame(datasets[0]['integrations_names'].map(lambda x: x.split(', ') if type(x) == str else []).
to_list()).stack().reset_index(level=1, drop=True)).groupby(level=0).sum()
#creating new df with dummy columns
df_dum = datasets[0].copy()
df_dum[df_dummies.columns] = df_dummies
df_dum=df_dum.drop(columns='integrations_names')
df_dum.columns
df1=df_dum.copy() #datasets[0]
df2 = df1.set_index('account_id')

df3 = df2.drop(columns='paid_status')
df3.fillna(0,inplace=True)
x_kmean = StandardScaler().fit_transform(df3) # need to standard scale the data

df3.columns

# determination of the number of clusters - the number should be chosen based on the 'knee bend'
from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(x_kmean)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

# It seems that the optimal number of clusters is five
km = KMeans(n_clusters=5)
clusters = km.fit_predict(x_kmean)
df3["label"] = clusters

df3.head()

plt.scatter(df3['dau_avg_last_30_days'],df3['seats'], c =df3['label'] )

import seaborn as sns
#figure, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
#figure.subplots_adjust(hspace = 1, wspace=.3)

sns.countplot(clusters).set_title('Customer clusters')
plt.show()
pass
#sns.countplot(clusters, ax=axs[1])
#axs[1].set_title('Azdias clusters')

df3[['account_age','seats','dau_avg_last_30_days']][df3['label']==4].describe()

df3[['account_age','seats','dau_avg_last_30_days']][df3['label']==0].describe()



