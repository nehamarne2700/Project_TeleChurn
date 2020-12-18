class customer:
    def __init__(self):
        self.__customerId=""
        self.__gender=""
        self.__seniorCitizen=""
        self.__partner=""
        self.__dependents=""
        self.__tenure=""
        self.__phoneService=""
        self.__multipleLines=""
        self.__internetService=""
        self.__onlineSecurity=""
        self.__onlineBackup=""
        self.__deviceProtection=""
        self.__techSupport=""
        self.__streamingTV=""
        self.__streamingMovies=""
        self.__contract=""
        self.__paperlessBilling=""
        self.__paymentMethod=""
        self.__monthlyCharges=0
        self.__totalCharges=0
        self.__churn=0
        
    def __init__(self,customerId, gender, seniorCitizen, partner, dependents,tenure, phoneService, multipleLines, internetService,onlineSecurity, onlineBackup, deviceProtection, techSupport,streamingTV, streamingMovies, contract, paperlessBilling,paymentMethod, monthlyCharges, totalCharges,churn):
        self.__customerId=customerId
        self.__gender=gender
        self.__seniorCitizen=seniorCitizen
        self.__partner=partner
        self.__dependents=dependents
        self.__tenure=tenure
        self.__phoneService=phoneService
        self.__multipleLines=multipleLines
        self.__internetService=internetService
        self.__onlineSecurity=onlineSecurity
        self.__onlineBackup=onlineBackup
        self.__deviceProtection=deviceProtection
        self.__techSupport=techSupport
        self.__streamingTV=streamingTV
        self.__streamingMovies=streamingMovies
        self.__contract=contract
        self.__paperlessBilling=paperlessBilling
        self.__paymentMethod=paymentMethod
        self.__monthlyCharges=monthlyCharges
        self.__totalCharges=totalCharges
        self.__churn=churn

    @property
    def customerId(self):
        return self.__customerId

    @property
    def gender(self):
        return self.__gender

    @property
    def seniorCitizen(self):
        return self.__seniorCitizen

    @property
    def partner(self):
        return self.__partner

    @property
    def dependents(self):
        return self.__dependents

    @property
    def tenure(self):
        return self.__tenure

    @property
    def phoneService(self):
        return self.__phoneService

    @property
    def multipleLines(self):
        return self.__multipleLines

    @property
    def internetService(self):
        return self.__internetService

    @property
    def onlineSecurity(self):
        return self.__onlineSecurity

    @property
    def onlineBackup(self):
        return self.__onlineBackup

    @property
    def deviceProtection(self):
        return self.__deviceProtection

    @property
    def techSupport(self):
        return self.__techSupport

    @property
    def streamingMovies(self):
        return self.__streamingMovies

    @property
    def streamingTV(self):
        return self.__streamingTV

    @property
    def contract(self):
        return self.__contract

    @property
    def paperlessBilling(self):
        return self.__paperlessBilling

    @property
    def paymentMethod(self):
        return self.__paymentMethod

    @property
    def monthlyCharges(self):
        return self.__monthlyCharges

    @property
    def totalCharges(self):
        return self.__totalCharges

    @property
    def churn(self):
        return self.__churn


#a=customer('a1','','','','','','','','','','','','','','','','','',0,0)
#print(a.customerID)