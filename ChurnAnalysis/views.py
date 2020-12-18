from django.shortcuts import render
from django.http import HttpResponse
from .apps import ChurnanalysisConfig
from .load import customer
import os 
import numpy as np 
import pandas as pd  
import matplotlib
matplotlib.use(('Agg'))
import matplotlib.pyplot as plt 
import sklearn
import seaborn
import PIL
from PIL import Image
from io import StringIO
from matplotlib import pylab
from pylab import *
import io
# Create your views here.

path=os.path.join(os.path.dirname(__file__),'churn.csv')
Tdata = pd.read_csv(path)
Tdata.TotalCharges= pd.to_numeric(Tdata.TotalCharges, errors='coerce')
Tdata['TotalCharges'].fillna((Tdata['TotalCharges'].mean()), inplace=True)
'''Tdata.drop('customerID',axis=1, inplace=True)
Num_cols = Tdata.select_dtypes(include=['float64','int64']).columns.tolist()
Cat_cols = Tdata.select_dtypes(include=['object']).columns.tolist()

#print(Tdata[Num_cols[1]])
n,bins,patches=plt.hist(Tdata[Num_cols[1]],5,facecolor='blue',alpha=0.5)
plt.savefig('hist.png')'''


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
    #print(Tdata.loc[Tdata['customerID']==val1].index.values)
    if(val1=="admin" and val2=="admin"):
        return render(request,'admin.html',{'name':val1})
    elif (len(Tdata.loc[Tdata['customerID']==val1].index.values))>0:
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
    predt=ChurnanalysisConfig.model.predict([data])[0]
    print(predt)
    cust=customer(cid, gender, senior, partner, dependents,tenure, service, mlines, iservice,olsecurity, olbackup, protection, tsupport,stv, smovie, contract, billing,pmethod, mcharge, tcharge,predt)
    return render(request,'predResult.html',{'cust':cust})

def graph(request):
    return render(request,'graph.html')
    

'''
def home(request):
    return render(request,'home.html',{'name':'Neha'})

def add(request):
    val1=int(request.POST["num1"])
    val2=int(request.POST["num2"])
    res = val1 + val2
    return render(request,'result.html',{'result':res})

'''

