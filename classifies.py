import random
import math
import NaiveBayes
import Perceptron
import NeuralNet


from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

def getTime():
    now = datetime.now()
    
    return now
#file - is the file to read
#version - is either "digit" or "face"
def readInput(file, fileType):
    if fileType == "digit":
        rows, cols = (28,28)
    elif fileType == "test":
        rows, cols = (2,2)
    else:
        rows, cols = (70,60)
    NNInput = [[0 for x in range(cols)] for y in range(rows)] 
    for i in range(rows): 
        file_line = file.readline()
        if file_line == False:
            return -1
        file_line= file_line.rstrip()
        for j in range(len(file_line)):
            if file_line[j] == "+":
                NNInput[i][j] = 1
            if file_line[j] == "#":
                NNInput[i][j] = 2
            if file_line[j] == " ":
                NNInput[i][j] = 0
    return NNInput        


def readInputResult(file):
    line = file.readline()
    line = line.rstrip()
    if line == False or line == "":
        return -1
    return line
        
#change this to run the diffrent files

if __name__ == "__main__":
    start = getTime()
    print(start)
    '''
    file_train = 'facedatatrain'
    file_train_results = 'facedatatrainlabels'
    file_test = 'facedatatest'
    file_test_results = 'facedatatestlabels'
    fileType = "face"  # can be Digit or Face
    '''
    file_train = 'trainingimages'
    file_train_results = 'traininglabels'
    file_test = 'testimages'
    file_test_results = 'testlabels'
    fileType = "digit"  # can be Digit or Face
    
    
    file_train = open(file_train, "r")
    file_train_results = open(file_train_results, "r")
    
    file_test = open(file_test, "r")
    file_test_results = open(file_test_results, "r")
    
    '''
    file_train.close()
    file_train_results.close()
      
    file_test.close()
    file_test_results.close()
    '''
    amountofdata = 1
    prob,count = NaiveBayes.runNaiveBayesClassifier(file_train,file_train_results,fileType,amountofdata)
    
    
    NaiveBayes.testNaiveBayesClassifier(file_test,file_test_results,prob,count,fileType)
    
    
    #Perceptron.startPerceptron(file_train,file_train_results,file_test,file_test_results,fileType)
    
    
    print(getTime()-start)
    
        #run the neural Net
    if fileType == "digit":
        nu = 28*28
        re = 10
    elif fileType == "test":
        nu = 4
        re = 1
    else:
        nu = 70*60
        re = 1
    '''
    NN = testfile.CreateNN([nu,2,2,re])

    NN = testfile.trainNN(NN,file_train,file_train_results,fileType, 10, 2)
    '''
    #nn = NeuralNetwork([nu,2,2,re])
    
    #for i in range(len(NN)):
        #for j in range(len(NN[i])):
            #print(NN[i][j].weights[0])
        #print(" ")
        

    file_train.close()
    file_train_results.close()
    
    file_test.close()
    file_test_results.close()
    
