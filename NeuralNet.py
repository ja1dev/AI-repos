import math
import classifies
import random

#creates a node object for the neural Network
class node:
    def __init__(self, weightLength):
        self.weights = [0]*weightLength
        self.bias = 1
        self.value = 0
        self.delta = 0
        self.error = 0
    def updateBias(self, bias):
        self.bias = 1
    def updateWeight(self, weight, value):
        self.weights[weight] = value
    def updatevalue(self, value):
        self.value = value
        
        
#A function that creates the arrays which store the nodes of each layer
#sets them to random weights and bias to start
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def Dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def costf(results, target):
    cost = results-target
    return cost

def trainNN(NN, file_test, file_results,fileType):
    text = 0    
    goal = 0

    outcome = [0,0]
    while (text != -1)&(goal != -1):
        text = classifies.readInput(file_test,fileType)
        goal = int(classifies.readInputResult(file_results))
        if (text != -1)&(goal != -1):
            if fileType == 'digit':
                result = [0]*10
            else:
                result = [0,0]
            result[goal] = 1    
            answer = runNN(NN,text)
            answer = int(answer.index(max(answer)))

            for i in range(len(NN)-1):
                cost = costf(result[i],answer)
            
            NN = BackProp(NN,answer,cost,result,text)
            

    return NN

def BackProp(NN,results,cost,target,inputs):
    #print(target)
    LL = 1
    data = []
    
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            data.append(inputs[i][j])
            
            
    Weighthold = []
    for i in range(len(NN)):
        Weighthold.append([])
        for j in range(len(NN[i])):
             Weighthold[i].append([])         
    for i in range(len(NN)): 
        for j in range(len(NN[i])):
            for k in range(len(NN[i][j].weights)):
                Weighthold[i][j].append(NN[i][j].weights[k])
                           

    for i in reversed(range(len(NN))):
        for j in range(len(NN[i])):
            for k in range(len(NN[i][j].weights)-1):
                if (i == 0):
                    deltaE = (NN[i][j].value-target[j])*Dsigmoid(NN[i][j].value)*data[k]
                else:
                    deltaE = (NN[i][j].value-target[j])*Dsigmoid(NN[i][j].value)*NN[i-1][j].value
                NN[i][j].weights[k] = NN[i][j].weights[k] - LL * deltaE
    
                '''
                if i == len(NN)-1:
                    NN[i][j].delta = (target[j]-NN[i][j].value)
                    #print("reeeee " + str(NN[i][j].delta))
                else:
                    for p in range(i,len(NN)):
                        
                    #NN[i][j].delta = NN[i][j].delta + NN[i+1][j].delta*NN[i+1][j].weights[k]
                        something = 7
                    
                    
                    
                #print("delta " + str(NN[i][j].delta))
                NN[i][j].delta  = NN[i][j].delta * Dsigmoid(NN[i][j].value)
                #print("change " + str(NN[i][k].value*NN[i][j].delta*LL))
                #print(len(NN[i][j].weights))
                #print("test" + str(NN[i][j].value))
                NN[i][j].updateWeight(k,NN[i][j].value*NN[i][j].delta*LL + NN[i][j].weights[k])
                '''
                ddeltaE = (NN[i][j].bias-target[j])*Dsigmoid(NN[i][j].bias)
                NN[i][j].weights[len(NN[i][j].weights)-1] = NN[i][j].bias - LL * deltaE
                

    return NN


def createNN (layers):
    NN = []
    for i in range(1,len(layers)):
        layer = []
        for j in range(layers[i]):

            nodes = node(layers[i-1]+1)
            nodes.updateBias(1)
            for k in range(len(nodes.weights)):
                nodes.updateWeight(k, 1)
            layer.append(nodes)
        NN.append(layer)

    #number of inputs
    #number of nodes per layer
    #number of layers
    #numberof outpu
    return NN
    
#runs the neural Netwrok with the given inputs
def runNN(NN,inputs):
    
    
    data = []
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            data.append(inputs[i][j])
    #print(len(NN[0]))
    for i in range(len(NN[0])):
        
        value = 0
        for j in range(len(data)):
            value = value + NN[0][i].weights[j]* data[j]
        value = value + NN[0][i].bias
        value = sigmoid(value)
        NN[0][i].updatevalue(value)
    
                    
                
    for i in range(1,len(NN)):
        for j in range(len(NN[i])):
            value = 0
            for k in range(len(NN[i][j].weights)-1):
                value = value + NN[i][j].weights[k]*NN[i-1][k].value
            value = value + NN[i][j].weights[len(NN[i][j].weights)-1]
            value = sigmoid(value)
            NN[i][j].updatevalue(value)
    ''' 
    Weighthold = []
    for i in range(len(NN)):
        Weighthold.append([])
        print(Weighthold)
        for j in range(len(NN[i])):
            Weighthold[i].append([])         
    print(Weighthold)
    for i in range(len(NN)): 
        for j in range(len(NN[i])):
                #print("Yvalue " + str(i)+ " "+str(j)+ " "+str(k))
                Weighthold[i][j].append(NN[i][j].value)
    print(Weighthold)
    '''
    output = []
    for i in range(len(NN[len(NN)-1])):
        output.append(NN[len(NN)-1][i].value)
        print(NN[len(NN)-1][i].value)
    return output
   
def testNN(NN, file_test, file_results, fileType):
    text = 0    
    goal = 0
    outcome = [0,0]
    while (text != -1)&(goal != -1):
        text = classifies.readInput(file_test,fileType)
        goal = int(classifies.readInputResult(file_results))
        if (text != -1)&(goal != -1):
            if fileType == 'digit':
                result = [0]*10
            else:
                result = [0,0]
            result[goal] = 1    
            answer = runNN(NN,text)
            #print(answer)
            answer = int(answer.index(max(answer)))
            #print(answer)
            
            if (answer == goal):    
                outcome[0] += 1
            else:    
                outcome[1] += 1
    print("results")
    print("correct: " + str(outcome[0]/sum(outcome)*100) + "%" )
    print("incorrect: " + str(outcome[1]/sum(outcome)*100) + "%" )
    
    
    


