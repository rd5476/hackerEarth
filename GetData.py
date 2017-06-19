import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import hstack
from blaze.expr.reductions import min
from sklearn.feature_extraction.text import TfidfVectorizer
def readFile(fileName):
    allData = []
    headers = []
    counter =0
    try:
        with open(fileName,encoding='utf8') as csvfile:

            fileReader = csv.reader(csvfile)
            for row in fileReader:
                headers = str(row)
                counter+=1
                break

            # for row in fileReader:
            #     temp = {}
            #     idx =0
            #     for r in row:
            #         temp[headers[idx]]=r
            #         idx+=1
            #     allData.append(temp)
            #     counter += 1
            for row in fileReader:
                temp = []
                idx =0
                for r in row:
                    temp.append(r)
                    #idx+=1
                allData.append(temp)
                counter += 1
    except Exception as e:

        print( e )
        print(counter)
    # for i in allData:
    #     for x in i:
    #         print(i[x])
    #     print('smaple over')
    return headers,allData


def textFeatures(descriptions):
    tfidf = TfidfVectorizer(min_df=1, stop_words='english',max_features=20)
    trainingData = tfidf.fit_transform(descriptions)
    #print(tfidf.get_feature_names())
    temp = trainingData.toarray()
    # print(temp[0])
    #
    # print(tfidf.inverse_transform(temp[0]))
    # print(temp[1])
    #
    # print(tfidf.inverse_transform(temp[1]))
    # idf = tfidf.idf_
    # print(dict(zip(tfidf.get_feature_names(), idf)))
    return temp
    pass

def convertCurrency(goal,currency):
    conversion = {}
    goals= []
    conversion['NOK'] = 0.12
    conversion['DKK'] = 0.15
    conversion['AUD'] = 0.76
    conversion['CAD'] = 0.75
    conversion['GBP'] = 1.28
    conversion['EUR'] = 1.12
    conversion['SEK'] = 0.13
    conversion['NZD'] = 0.73
    conversion['USD'] = 1
    for idx  in range(len(goal)):
        goals.append([conversion[currency[idx]]*goal[idx],conversion[currency[idx]]])

   # goals = np.asarray(goals)
    #print(type(goals))
    #goals = np.reshape(goals,(1,len(goals)))
    return goals

def convertBoolean(data):
    temp=[]
    for i in data:
        if i=='false' or i=='False' or i=='FALSE': data = temp.append([0])
        else: temp.append([1])
    return temp
def featureExtraction(fileName):

    headers,allData = readFile(fileName)
    #
    # with open('allData.pickle', 'wb') as handle:
    #     pickle.dump(allData, handle)
    # with open('headers.pickle', 'wb') as handle:
    #     pickle.dump(headers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('allData.pickle', 'rb') as handle:
    #     allData = pickle.load(handle)
    # with open('headers.pickle', 'rb') as handle:
    #     headers = pickle.load(handle)
    # allData = np.asarray(allData)
    # print('conversion done')
    # for i in allData:
    #     for x in i:
    #         print(i[x])
    #     print('smaple over')

    #print(len(allData))
    temp = np.asarray(allData)

    temp[:,8:14] = np.asarray(temp[:,8:14],dtype='int')
    #print(temp[1:4,2])
    #print( np.asarray(temp[:,8:14],dtype='int'))
    textFeat = textFeatures(temp[:,2])
    #print(np.shape(textFeat))
    #curreny = set(temp[:,7])
    #print(curreny)
    disComm = convertBoolean(temp[:,5])
    goal = convertCurrency(np.asarray(temp[:,3],dtype='float'),temp[:,7])
    #print(goal)
    #print(len(goal),goal)
    #goal = np.asarray(goal,dtype='float')
    #print(goal)
    allFeatures =np.asarray( np.hstack((textFeat,goal,disComm)),dtype='float16')
    print(allFeatures.shape)
    allFeatures = np.concatenate((allFeatures,temp[:,8:13]),axis=1)
    y_train = np.asarray(temp[:,13],dtype='int')

    return  allFeatures,y_train
if __name__=='__main__':

    rf = RandomForestClassifier(n_estimators=30)

    allFeatures,labels = featureExtraction('train.csv')
    #print(allFeatures)
    model = rf.fit(allFeatures,labels)

    testFeatures, testLabel = featureExtraction('test.csv')
    result = rf.predict(testFeatures)
    count =0
    for idx in range(len(testFeatures)):
        if testLabel[idx]== result[idx]:
            count+=1
    print(count)
    print(count/len(testLabel))
    # print(allFeatures)
    # print(labels    )