import openml as oml
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import csv
import json
import simple_pickle
import pandas as pd
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
import numpy as np
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt

#columns = ['did','name','NumberOfInstances', 'NumberOfFeatures','NumberOfClasses']
#data_dict = oml.datasets.list_datasets(offset = 2000) # Returns a dict
#data_list = pd.DataFrame.from_dict(data_dict, orient='index') # dataframe
def CreateFile(filename):
    f = open(filename + ".txt", "w+")
    f.close()

def Evaluate(id,filename):
    # https://www.openml.org/t/167140
    metric = 'root_mean_squared_error'
    list = []
    evals = oml.evaluations.list_evaluations(function=metric, task=[id])
    #for a in evals:
    #    s = str(a) + ": "
    #    f.write(s)
    #    store = str(vars(evals[a]))
    #    for char in store:
    #        if (char == ',') or (char =='}'):
    #            f.write('\n')
    #        if (char != ',') and (char != "'") and (char != ' ') and (char != '}'):
    #            f.writelines(char)

    #Make a dataframe with columns = keys in dictionary
    #For every dictionary, add a new row to dataframe
    f = open(filename +".txt","a+")
    f.write(str(id)+":\n")
    for a in evals:
        dictionary = vars(evals[a])
        Dframe = pd.DataFrame.from_dict(dictionary,orient = 'index')
        list.append(Dframe)
        for item in list:
            f.write("%s\n"%item)
    f.close()
   # print(Dframe.head(20))

def restore(list1,list2):
    for i in list1:
        list2.append([])
    return list2


def extract_rows(filename):
    res = pd.DataFrame(columns = ['data.id','name','version','status','format','tags','majority.class.size','max.nominal.att.distinct.values','minority.class.size','number.of.classes','number.of.features','number.of.instances','number.of.missing.values','number.of.numeric.features','number.of.symbolic.features'])
    df = pd.read_csv(filename + '.csv')
    f = open("filtered_dataset.csv", "a+")
    df.to_csv("filtered_dataset.csv")
    f.close()

    array = [23380, 4550, 23512, 1176, 1112, 1114, 1476, 40509, 1459, 40496, 40536, 1486, 1590, 4134, 4538, 1478, 23381, 1485, 1461, 40499, 40668, 4534, 300, 1036, 1038, 1053, 188, 42, 38, 29, 451, 15, 2, 554, 470, 44, 46, 37, 23, 54, 28, 32, 36, 50, 1067, 1220, 1120, 1068, 1046, 1043, 1050, 1063, 1049, 478, 469, 1468, 1466, 1467, 1464, 1462, 1471, 4135, 1570, 312, 151, 182, 183, 60, 1501, 1504, 1494, 1497, 1510, 1505, 1493, 377, 458, 335, 334, 333, 1515, 20, 21, 12, 14, 6, 11, 16, 18, 3, 1480, 1479, 1487, 1489, 1491, 1492, 1475, 307, 6332, 24, 375, 22, 31]
    for i in array:
        res.append(df.loc['data.id'] == i)







def Store_as_array(filename):
    df = pd.read_csv(filename + '.csv')
    data_list = df.T.values.tolist()
    flow_name = data_list[5]
    data_name = data_list[9]
    accuracy = data_list[11]
    filtered_flow = []
    filtered_data = []
    filtered_accuracy = []
    idx = []
    temp = []
    counter = 0
    accum = 0
    flag1 = True
    flag2 = True
    data = []
    templist = []
    columns_list = ["Flow name"]
    for i in range (0,len(data_name)):
        for x in filtered_data:
            if (data_name[i] == x):
                flag1 = False
        for y in filtered_flow:
            if (flow_name[i] == y):
                flag2 = False
        if (flag1 == True):
            filtered_data.append(data_name[i])
        if (flag2 == True):
            filtered_flow.append(flow_name[i])
        flag1 = True
        flag2 = True

    for i in filtered_data:
        filtered_accuracy.append([])
        idx.append([])

    restore(filtered_flow,temp)

    for i in range(0,len(filtered_data)):
        for x in range(0,len(accuracy)):
            if (data_name[x] == filtered_data[i]):
                idx[i].append(x)

    for i in range (0,len(idx)):
        for q in idx[i]:
            for k in range(0,len(filtered_flow)):
                if (flow_name[q] == filtered_flow[k]):
                    temp[k].append(accuracy[q])

        for p in temp:
            for s in p:
                counter += 1
                accum += s
            if (counter != 0):
                mean = accum / counter
                filtered_accuracy[i].append(mean)
            elif (counter == 0):
                filtered_accuracy[i].append(0)
            counter = 0
            accum = 0
        temp = []
        restore(filtered_flow,temp)


    for i in range (0,len(filtered_flow)):
        templist += [filtered_flow[i]]
        for q in range(0,len(filtered_data)):
            templist += [filtered_accuracy[q][i]]
        data += [templist]
        templist = []

    for i in filtered_data:
        columns_list += [i]

    df = pd.DataFrame(data, columns=columns_list)
    f = open("STUDY_14_RMSE.csv", "a+")
    df.to_csv("STUDY_14_RMSE.csv")
    f.close()



def predict():
    rating = Store_as_array("botV1RMSE1")
    rating = rating.drop(columns = "Learner name")
    U, s, VT = svd(rating)
    Sigma = zeros((rating.shape[0], rating.shape[1]))
    Sigma[:rating.shape[0], :rating.shape[0]] = diag(s)
    B = U.dot(Sigma.dot(VT))
    df = pd.DataFrame(B)
    f = open("PREDICTION.csv", "a+")
    df.to_csv("PREDICTION.csv")
    f.close()






def convert(filename):
    c = open(filename + ".txt","r")
    list1 = []
    list2 = []
    list3 = []
    flowList = []
    dataList = []
    count = 0
    valueList = []
    r = c.readlines()
    for i in r:
        count += 1
        if ((count-6)%13 == 0):
            list1 += [i]
        elif ((count-8)%13 == 0):
            list2 += [i]
        elif ((count-11)%13 == 0):
            list3 += [i]
    for x in list1:
        flowname = x.lstrip("flow_name")
        flowname = flowname.lstrip()
        flowList += [flowname]

    for x in list2:
        dataname = x.lstrip("data_name")
        dataname = dataname.lstrip()
        dataList += [dataname]

    for x in list3:
        value = x.lstrip("value")
        value = value.lstrip()
        valueList += [value]

    data = []
    for i in range(0,399462):
        data += [[flowList[i],dataList[i],valueList[i]]]

    df = pd.DataFrame(data,columns=["flow name","data name","value"])
    #CreateFile("Supervised Classification(Precision)")
    f = open("Supervised Classification(Precision).csv", "a+")
    df.to_csv("Supervised Classification(Precision).csv")
    #list = []
    #list.append(df)
    #for item in list:
    #        f.write("%s\n"%item)
    f.close()



Store_as_array("study_14_rmse_15July")

#convert("Test")
#Store_as_array("botV1RMSE1")
#extract_rows("datasetfeatures")

#task_list = [1765, 1766, 1767, 1769, 1770, 1771, 1772, 1773, 1774, 1775]
#for i in task_list:
   # Evaluate(i,"Test_2")

#Read_From_File("Test")








