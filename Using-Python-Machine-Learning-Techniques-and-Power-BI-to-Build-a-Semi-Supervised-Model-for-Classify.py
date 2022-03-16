import pandas as pd
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#When Running in the Power BI after the dataset is loaded on the Power BI change
# dataset = dataset

dataset = pd.read_excel("D:\\Disk_Drive\\DocumentsBuckUp\\360DigiDataScience\\Project\\Performance.xlsx")
dataset = dataset.dropna()

# Add a new map if new disgnation is going to be added this is to show the pairplot and heat map corrlation
dataset_designation = dataset['Designation'].map({'Business Analyst': 8,	'Data Engineer': 7,	'Data Analyst': 6,	'Data Scientist': 5,	'Domain Expert': 4,	'Chief Data Scientist': 3,	'Software Developer': 2,	'Database Administrator': 1,	'Data Architect': 0})
dataset_promotion = dataset['Promotion'].map({'No':0, 'Yes':1})
dataset_personlity = dataset['Personality_trait_score'].map({'Bad':0, 'Good':1, 'Best':2})

dataset_designation = pd.DataFrame(dataset_designation)
dataset_promotion = pd.DataFrame(dataset_promotion)
dataset_personlity = pd.DataFrame(dataset_personlity)

#Normalization of Data using Min Max
def minmax(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

dataset_scale = pd.concat([dataset.iloc[:,4:6], dataset.iloc[:,7:9]],axis=1)
dataset_minmax = pd.DataFrame(minmax(dataset_scale), columns=(dataset_scale.columns))

new_info = pd.concat([dataset_minmax, dataset_designation, dataset_promotion, dataset_personlity], axis = 1)
new_info_data = pd.DataFrame(new_info.iloc[:,2:4])

# sns.pairplot(new_info)
# sns.heatmap(new_info.corr(), annot=True)

TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_info_data)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
# plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("new_info")

kmeans = KMeans(n_clusters = 3, max_iter=300, verbose=1)
kmeans.fit(new_info_data)

identified_cluster = kmeans.fit_predict(new_info_data)

dataset_cluster = dataset.copy()
dataset_cluster['means_cluster'] = identified_cluster

'''
sns.scatterplot(dataset_cluster['MonthlySalary'], dataset_cluster['Openess_to_experience'], hue="means_cluster", 
                data=dataset_cluster, palette='rainbow', s=100)
'''

#Saving the cluster in excel formant
# dataset_cluster.to_excel (r'C:\\Users\\Ashner_Novilla\\Documents\\360DigiDataScience\\Project\\dataset_cluster.xlsx', index = False, header=True)


## Performing the supervised Learning Algorith
# Using the dataset_cluster.xlsx as the training uncomment the below method
# supervised_data_promotion = pd.read_excel('C:\\Users\\Ashner_Novilla\\Documents\\360DigiDataScience\\Project\\dataset_cluster.xlsx')
supervised_data_promotion = dataset_cluster['Promotion'].map({'No':0, 'Yes':1})
supervised_data_onehot = pd.get_dummies(data=dataset_cluster, columns=['Designation', 'Improvement_in_ExamScore', 'Working_hours', 'Personality_trait_score'])
supervised_data = pd.concat([supervised_data_promotion, supervised_data_onehot.iloc[:,5:], pd.DataFrame(minmax( dataset_cluster.iloc[:,7:9]))], axis=1)

#independed variable and target
supervised_data_x = supervised_data.loc[:, supervised_data.columns != 'means_cluster']
supervised_data_y = supervised_data.loc[:, supervised_data.columns == 'means_cluster']

from sklearn.model_selection import train_test_split
train_x, text_x = train_test_split(supervised_data_x, test_size = 0.3, random_state = 123)
train_y, text_y = train_test_split(supervised_data_y, test_size = 0.3, random_state = 123)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

#Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

#Logistic Regression
logitmodel = LogisticRegression(multi_class = "multinomial", solver = "saga").fit(train_x, train_y)
print('Accuracy of multi classifier on training set: {:.2f}'.format(logitmodel.score(train_x, train_y)))
print('Accuracy of multi Tree classifier on test set: {:.2f}'.format(logitmodel.score(text_x, text_y)))

# Decision Tree
clf = DecisionTreeClassifier(max_depth=3).fit(train_x, train_y)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(train_x, train_y)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(text_x, text_y)))

#KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_x, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(train_x, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(text_x, text_y)))

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x, train_y)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(train_x, train_y)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(text_x, text_y)))

mnb = MultinomialNB()
mnb.fit(train_x, train_y)
print('Accuracy of MNB classifier on training set: {:.2f}'.format(mnb.score(train_x, train_y)))
print('Accuracy of MNB classifier on test set: {:.2f}'.format(mnb.score(text_x, text_y)))

#Random Forrest
rf = RandomForestClassifier(n_jobs=8, n_estimators=2, criterion="entropy", random_state=96)
rf.fit(train_x, train_y)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(rf.score(train_x, train_y)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(text_x, text_y)))

#AdaBoost
ada_clf = AdaBoostClassifier(learning_rate = 0.002, n_estimators = 100)
ada_clf.fit(train_x, train_y)
print('Accuracy of Ada Boost classifier on training set: {:.2f}'.format(ada_clf.score(train_x, train_y)))
print('Accuracy of Ada Boost classifier on test set: {:.2f}'.format(ada_clf.score(text_x, text_y)))

#XGBoost
xb = XGBClassifier(n_estimators=1)
xb.fit(train_x, train_y)
print('Accuracy of XG Boost classifier on training set: {:.2f}'.format(xb.score(train_x, train_y)))
print('Accuracy of XG Boost classifier on test set: {:.2f}'.format(xb.score(text_x, text_y)))

xgb_clf = XGBClassifier(max_depths = 1, n_estimators = 1, learning_rate = 0.0002, n_jobs = -10)
xgb_clf.fit(train_x, train_y)
print('Accuracy of XG Boost classifier on training set: {:.2f}'.format(xgb_clf.score(train_x, train_y)))
print('Accuracy of XG Boost classifier on test set: {:.2f}'.format(xgb_clf.score(text_x, text_y)))

## Base on Mdel Selection Best Fit is RF and DT  - the developer will go with DT due to simpleness of algorithm 
# and the accuracy does not significantly change with RF
### Predicting values using DT
identified_target = pd.DataFrame(clf.predict(supervised_data_x))

dataset_final = dataset_cluster.copy()
dataset_final['Predicted'] = identified_target
#Saving the dataset final
#dataset_final.to_excel (r'D:\\Disk_Drive\\DocumentsBuckUp\\360DigiDataScience\\Project\\dataset_final.xlsx', index = False, header=True)
from sklearn.metrics import accuracy_score
accuracy_score(supervised_data_y, identified_target)

dataset_final['match'] = np.where(dataset_final['means_cluster']==dataset_final['Predicted'], 1,0)


'''
#This is for additional data test only
# New additional data for independent variables - change the data location and file name following the below format for the new testing the data.
new_supervised_dataset = pd.read_excel('C:\\Users\\Ashner_Novilla\\Documents\\360DigiDataScience\\Project\\new_supervised_data.xlsx')
#data wrangling for the new set of data
new_supervised_data_promotion = new_supervised_dataset['Promotion'].map({'No':0, 'Yes':1})
new_supervised_data_onehot = pd.get_dummies(data=new_supervised_dataset, columns=['Designation', 'Improvement_in_ExamScore', 'Working_hours', 'Personality_trait_score'])
new_supervised_data = pd.concat([supervised_data_promotion, supervised_data_onehot.iloc[:,5:], pd.DataFrame(minmax( dataset_cluster.iloc[:,7:9]))], axis=1)

#Splitting the train and tes data
new_supervised_data_x = new_supervised_data.loc[:, new_supervised_data.columns != 'means_cluster']

#predicting the new data
new_supervised_data_predict = pd.DataFrame(clf.predict(new_supervised_data_x))
#Concatinating the new set of data to the 
new_supervised_dataset["new_supervised_data_value"] = new_supervised_data_predict
# new_supervised_dataset.to_excel (r'C:\\Users\\Ashner_Novilla\\Documents\\360DigiDataScience\\Project\\new_supervised_dataset_final.xlsx', index = False, header=True)
'''
