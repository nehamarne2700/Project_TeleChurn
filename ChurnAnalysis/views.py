from django.shortcuts import render
from django.http import HttpResponse
from .apps import ChurnanalysisConfig
from .load import customer
import os 
import numpy as np 
import pandas as pd  
import matplotlib
matplotlib.use(('Agg'))
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, make_scorer, accuracy_score, roc_curve, confusion_matrix, classification_report


from matplotlib import pylab
from pylab import *
import io
# Create your views here.

path=os.path.join(os.path.dirname(__file__),'churn.csv')
Tdata1 = pd.read_csv(path)
Tdata1.TotalCharges= pd.to_numeric(Tdata1.TotalCharges, errors='coerce')
Tdata1['TotalCharges'].fillna((Tdata1['TotalCharges'].mean()), inplace=True)
Tdata =pd.read_csv(path)

Tdata1.drop('customerID',axis=1, inplace=True)
#print(Tdata.head())
Num_cols = Tdata1.select_dtypes(include=['float64','int64']).columns.tolist()
Cat_cols = Tdata1.select_dtypes(include=['object']).columns.tolist()
#print(Tdata1[Num_cols[1]])
Tdata1[Num_cols].hist(figsize = (10,10))
plt.savefig('static/img/graphs/hist.png',bbox_inches="tight")
plt.clf()
plt.figure(figsize=(10,10))
sns.set(font_scale=2)
sns.set_style("whitegrid")
sns_plot=sns.countplot(x="Churn", hue="gender", data=Tdata1)
plt.xlabel("Churn",size=23)
plt.ylabel("Gender",size=23)
plt.title("Gender vs Churn ",size=23)
plt.savefig('static/img/graphs/gender.png',bbox_inches="tight")

Binary_class = Tdata1[Cat_cols].nunique()[Tdata1[Cat_cols].nunique() == 2].keys().tolist()
Multi_class =  Tdata1[Cat_cols].nunique()[Tdata1[Cat_cols].nunique() > 2].keys().tolist()

cr=['b','r','g']
sns.set(font_scale=1)
sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
for i, item in enumerate(Multi_class):
    if i < 3:
        ax = Tdata1[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0,color=cr)
        
    elif i >=3 and i < 6:
        ax = Tdata1[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0,color=cr)
        
    elif i < 9:
        ax = Tdata1[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0,color=cr)
    #ax.grid(False)
    ax.set_title(item)

plt.savefig('static/img/graphs/subplot.png',bbox_inches="tight")

plt.clf()
sns.catplot(x="Partner", hue="Churn", col="Churn",data=Tdata1, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),
linewidth=5,edgecolor=sns.color_palette("dark", 5))
plt.savefig('static/img/graphs/partner.png',bbox_inches="tight")

plt.clf()
sns.catplot(x="Dependents", hue="Churn", col="Churn",data=Tdata1, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),
linewidth=5,edgecolor=sns.color_palette("dark", 5))
plt.savefig('static/img/graphs/dependents.png',bbox_inches="tight")

plt.clf()
sns.catplot(x="PhoneService", hue="Churn", col="Churn",data=Tdata1, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),
linewidth=5,edgecolor=sns.color_palette("dark", 5))
plt.savefig('static/img/graphs/phoneservice.png',bbox_inches="tight")

plt.clf()
sns.catplot(x="PaperlessBilling", hue="Churn", col="Churn",data=Tdata1, kind="count",height=4, aspect=.7,  facecolor=(0, 0, 0, 0),
linewidth=5,edgecolor=sns.color_palette("dark", 5))
plt.savefig('static/img/graphs/paperlessbilling.png',bbox_inches="tight")

#Label encoding Binary columns
le = LabelEncoder()
for i in Binary_class :
    Tdata1[i] = le.fit_transform(Tdata1[i])

# Split multi class catergory columns as dummies  
Tdata_Dummy = pd.get_dummies(Tdata1[Multi_class])

New_df = pd.concat([Tdata1[Num_cols],Tdata1[Binary_class],Tdata_Dummy], axis=1)
#print(New_df.shape)

# Data to plot Percent of churn  
labels =["No Churn","Churn"]
sizes = New_df['Churn'].value_counts(sort = True)

colors = ["whitesmoke","red"]
explode = (0.1,0)  # explode 1st slice
plt.clf()
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=270,radius=1.5)
plt.legend()
plt.title('Percent of churn in customer')
plt.savefig('static/img/graphs/churnGraph.png',bbox_inches="tight")

#correlation
corr = New_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
plt.clf()

sns.set(font_scale=1)
cmap=sns.light_palette("seagreen", reverse=True)
sns.set_style("whitegrid")
#plt.gcf().subplots_adjust(bottom=0.25)
sns_plot=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('static/img/graphs/correlation.png',bbox_inches="tight")

plt.clf()
plt.figure(figsize=(15,8))
New_df.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.savefig('static/img/graphs/correlationNew.png',bbox_inches="tight")

X = New_df.loc[:, New_df.columns != 'Churn']
y = New_df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state =1)
params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}

# Fit RandomForest Classifier
clf = RandomForestClassifier(**params)
clf = clf.fit(X, y)
# Plot features importances
imp = pd.Series(data=clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.clf()
sns.set_style("whitegrid")
plt.figure(figsize=(10,12))
plt.title("Feature importance")
ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_r", orient='h')
plt.savefig('static/img/graphs/featureImp.png',bbox_inches="tight")

print('The number of samples into the Train data is {}.'.format(x_train.shape[0]))
print('The number of samples into the Test data is {}.'.format(x_test.shape[0]))

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(x_train,y_train)
accuracy = logistic_model.score(x_test,y_test)
accuracy=accuracy*100
print("Logistic Regression accuracy is :",accuracy)

#for Logistic Regression
cm_lr = confusion_matrix(y_test,logistic_model.predict(x_test))
#confusion matrix visualization
plt.clf()
f, ax = plt.subplots(figsize = (5,5))
labels=["No Churn","Churn"]
sns.heatmap(cm_lr,xticklabels=labels,yticklabels=labels, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Logistic Regression")
plt.savefig('static/img/graphs/confusionMatrix.png',bbox_inches="tight")

#data=[0,1,29,29,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0]
#print(logistic_model.predict([data]))


def loadCSV(request):
    return render(request,'data.html',{'Tdata':Tdata})

def blank(request):
    return render(request,'blank.html')

def login(request):
    flag=bool
    flag=False
    return render(request,'login.html',{'flag':flag})

def main(request):
    val1=str(request.POST["username"])
    val2=str(request.POST["password"])
    print(Tdata.loc[Tdata['customerID']==val1].index.values)
    if(val1=="admin" and val2=="admin"):
        return render(request,'admin.html',{'name':val1})
    elif (len(Tdata.loc[Tdata['customerID']==val1].index.values))>0 and (len(Tdata.loc[Tdata['customerID']==val2].index.values))>0:
        index=Tdata.loc[Tdata['customerID']==val1].index.values[0]
        cust=customer(Tdata.iloc[index,0],Tdata.iloc[index,1],Tdata.iloc[index,2],Tdata.iloc[index,3],Tdata.iloc[index,4],Tdata.iloc[index,5], Tdata.iloc[index,6],Tdata.iloc[index,7], Tdata.iloc[index,8],Tdata.iloc[index,9], Tdata.iloc[index,10], Tdata.iloc[index,11], Tdata.iloc[index,12],Tdata.iloc[index,13], Tdata.iloc[index,14], Tdata.iloc[index,15], Tdata.iloc[index,16],Tdata.iloc[index,17],Tdata.iloc[index,18], Tdata.iloc[index,19],Tdata.iloc[index,20])    
        return render(request,'customer.html',{'cust':cust})
    else:
        flag=bool
        flag=True
        return render(request,'login.html',{'flag':flag})

def detailsForm(request):
    return render(request,'predForm.html')

def predict(request):
    name=request.POST["cname"]
    cid=request.POST["cid"]
    gender=request.POST["gender"]
    senior=request.POST["senior"]
    partner=request.POST["partner"]
    dependents=request.POST["dependents"]
    tenure=int(request.POST["tenure"])
    service=request.POST["service"]
    mlines=request.POST["mlines"]
    iservice=request.POST["iservice"]
    olsecurity=request.POST["olsecurity"]
    olbackup=request.POST["olbackup"]
    protection=request.POST["protection"]
    tsupport=request.POST["tsupport"]
    stv=request.POST["stv"]
    smovie=request.POST["smovie"]
    contract=request.POST["contract"]
    billing=request.POST["billing"]
    pmethod=request.POST["pmethod"]
    mcharge=int(request.POST["mcharge"])
    tcharge=int(request.POST["tcharge"])
    #print(name,gender)
    data=[]
    a,b=1,0
    data.append(a if senior=="yes" else b)
    data.append(tenure)
    data.append(mcharge)
    data.append(tcharge)
    data.append(a if gender=="male" else b)
    data.append(a if partner=="yes" else b)
    data.append(a if dependents=="yes" else b)
    data.append(a if service=="yes" else b)
    data.append(a if billing=="yes" else b)
    data.append(a if mlines=="no" else b)
    data.append(a if mlines=="no service" else b)
    data.append(a if mlines=="yes" else b)
    data.append(a if iservice=="dsl" else b)
    data.append(a if iservice=="fiber optic" else b)
    data.append(a if iservice=="no" else b)
    data.append(a if olsecurity=="no" else b)
    data.append(a if olsecurity=="no service" else b)
    data.append(a if olsecurity=="yes" else b)
    data.append(a if olbackup=="no" else b)
    data.append(a if olbackup=="no service" else b)
    data.append(a if olbackup=="yes" else b)
    data.append(a if protection=="no" else b)
    data.append(a if protection=="no service" else b)
    data.append(a if protection=="yes" else b)
    data.append(a if tsupport=="no" else b)
    data.append(a if tsupport=="no service" else b)
    data.append(a if tsupport=="yes" else b)
    data.append(a if stv=="no" else b)
    data.append(a if stv=="no service" else b)
    data.append(a if stv=="yes" else b)
    data.append(a if smovie=="no" else b)
    data.append(a if smovie=="no service" else b)
    data.append(a if smovie=="yes" else b)
    data.append(a if contract=="mtm" else b)
    data.append(a if contract=="1" else b)
    data.append(a if contract=="2" else b)
    data.append(a if pmethod=="echeck" else b)
    data.append(a if pmethod=="mcheck" else b)
    data.append(a if pmethod=="btransfer" else b)
    data.append(a if pmethod=="ccard" else b)
    print(data,size(data))
    #predt=ChurnanalysisConfig.model.predict([data])[0]
    predt=logistic_model.predict([data])[0]
    print(predt)
    cust=customer(cid, gender, senior, partner, dependents,tenure, service, mlines, iservice,olsecurity, olbackup, protection, tsupport,stv, smovie, contract, billing,pmethod, mcharge, tcharge,predt)
    return render(request,'predResult.html',{'cust':cust})

def graph(request):
    return render(request,'graph.html',{'accuracy':accuracy})
    
def help(request):
    return render(request,'help.html')
    
def admin(request):
    return render(request,'admin.html')

'''
def home(request):
    return render(request,'home.html',{'name':'Neha'})

def add(request):
    val1=int(request.POST["num1"])
    val2=int(request.POST["num2"])
    res = val1 + val2
    return render(request,'result.html',{'result':res})

'''

