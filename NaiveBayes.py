import classifies
import math
def createProbability(fileType):
    prob = []
    if fileType == "digit":
        rows, cols = (28,28)
        for i in range(10):
            prob.append([])
            for j in range(rows*cols):
                prob[i].append([1]*3)
            
    else:
        rows, cols = (70,60)
        for i in range(2):
            prob.append([])
            for j in range(rows*cols):
                prob[i].append([1]*3)
   
    return prob

#calculating the probabilities
#adds the the list to get a count of the times each symble has appeared in each position

def trainingNaiveBayesClassifier(inputs, result ,prob):
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            for k in range(3):
                if (k == inputs[i][j]):
                    prob[result][i*len(inputs[i])+j][k] += 1 
    return prob


def runNaiveBayesClassifier(file_text, file_result , fileType,amountofdata):
    if fileType == "digit":
        count = [1]*10
        tot = 500
    else:
        count = [1,1]
        tot = 45
    prob = createProbability(fileType)
    text = 0
    result = 0
    cnt = 0
    while (text != -1)&(result != -1)&(cnt < (tot*amountofdata*10)):
        text = classifies.readInput(file_text,fileType)
        result = classifies.readInputResult(file_result)
        if (text != -1)&(result != -1):
            result = int(result)
            count[result] +=  1
            prob = trainingNaiveBayesClassifier(text , result , prob)
        cnt = cnt+1
    print(cnt)
    #finding probabilities
    temp = sum(count)
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            for k in range(len(prob[i][j])):
                prob[i][j][k] = prob[i][j][k]/(count[i]+2)
            
    for i in range(len(count)):
        count[i] = count[i]/temp

    return prob,count

def NaiveBayesClassifier (inputs, prob,count):
    data = []
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            data.append(inputs[i][j])

    answer = [0]*len(prob)
    for i in range(len(prob)):
        answer[i] = count[i]
        for k in range(len(prob[i])):
            answer[i] = answer[i]+ math.log(prob[i][k][data[k]])
    return answer

def testNaiveBayesClassifier(file_text, file_result,prob,count,fileType):    
    text = 0    
    result = 0
    outcome = [0,0]
    while (text != -1)&(result != -1):
        text = classifies.readInput(file_text,fileType)
        result = int(classifies.readInputResult(file_result))
        if (text != -1)&(result != -1):
            answer = NaiveBayesClassifier(text,prob,count)
            answer = int(answer.index(max(answer)))
         
            if (answer == result):
               
                outcome[0] += 1
            else:
               
                outcome[1] += 1
    print("results")
    print(str(outcome[0])+"out of "+str(sum(outcome)))
    print("correct" + str(outcome[0]/sum(outcome)*100) + "%" )
