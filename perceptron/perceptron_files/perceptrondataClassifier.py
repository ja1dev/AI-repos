# Used code from the berkeley link assignment gave us
# this is comprised of mainly the dataClassifier.py (removed and added parts not needed for perceptron) and converted the code to python3

import perceptron
import samples
import util
import time

TEST_SET_SIZE = 3
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

#deleted the rest because I didn't need it

# Main harness code
def runClassifier():
  #ask user for data
  facesordigits= input('Faces or digits dataset?: ')
  trainingdatavalue = int(input('Training data #: '))
  testingdatavalue = int(input('Testing data #: '))

  #get datum pixels for perceptron and update legalLabels (taken from original dataClassifier in analysis function)
  if (facesordigits == "faces"):
    featureFunction = basicFeatureExtractorFace
    legalLabels = range(2)
  else:
    featureFunction = basicFeatureExtractorDigit
    legalLabels = range(10)
  
  # Load data  
  numTraining = trainingdatavalue
  numTest = testingdatavalue

  classifier = perceptron.PerceptronClassifier(legalLabels, TEST_SET_SIZE)
  if(facesordigits=="faces"):
    rawTrainingData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT,True)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", chosenList)
  else:
    rawTrainingData, chosenList = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT,True)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", chosenList)
    
  
  # Extract features
  print("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  testData = list(map(featureFunction, rawTestData))
  
  # Conduct training and testing
  #time the time it takes to train
  print("Training...")
  start = time.time()
  classifier.train(trainingData, trainingLabels)
  totaltime = time.time() - start
  print('Training time: ', totaltime)

  print("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print("Results: ", str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))

if __name__ == '__main__': 
  # Run classifier
  runClassifier()