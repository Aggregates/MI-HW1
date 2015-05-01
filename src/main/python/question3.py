#training parameters for support vector machines:
from svm import *

c = 0.9

gamma = 6


kernel = RBF

#include/import the libraries we need for loading CSV files -------------------------------------------------------------------------------------------------------------------


def loadCSV(filename,multiclass=True,outputs=1,separator=','):
    #read in all the lines
    f = open(filename).readlines()
    
    #start our datasets
    in_data = []
    out_data =[]
    
    #process the file
    for line in f:
        #remove whitespace and split according to separator character
        samples = line.strip(' \r\n').split(separator)
        
        #save input data
        in_data.append([float(i) for i in samples[:-outputs]])
        
        #save output data
        if multiclass:
            out_data.append((samples[-1]))
        else:
            out_data.append([float(i) for i in samples[-outputs:]])
        
    #process multiclass encoding
    keys = []
    keysall = []
    if multiclass:
        processed_out_data = []
        #get all the unique values for classes
        keys = []
        keysall = []
        for d in out_data:
            keysall.append(d)

        for d in out_data:
            if d not in keys:
                keys.append(d)
        keys.sort()
    
    #use libsvm's data container:
    return svm_problem([keys.index(i) for i in out_data],in_data),in_data,[keys.index(i) for i in out_data],keys, keysall
    


#train the SVM ---------------------------------------------------------------------------------------------------------------------------------------------------
#include the current path for library importing so that we can find the svm files
import sys
import os
from time import strftime, time
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path)

#set up more svm library stuff
from svmutil import svm_train,svm_predict
svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

#load data
dataset,data,outputs,keys, keysall = loadCSV("activity.csv")

#set parameters
parameters = svm_parameter()
parameters.kernel_type = kernel
parameters.C = c
parameters.gamma = gamma

print "Training..."

#train
start = time()
solver = svm_train(dataset,parameters)
end = time()

trainingTime = end - start
print trainingTime

#Save trained SVM
#--------------------------------------------------------------------------------------------------------------------------------------------
from svmutil import svm_save_model

uniqueFileName = "generated\\Q3-TrainedSVM-" + strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
#svm_save_model(uniqueFileName,solver)
#testIdx = int(0.2*len(keysall))
#p_lbl, p_acc, p_prob = svm_predict(keysall[testIdx:], data[testIdx:], solver)

#visualise final classifications --------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plot
classes = keys #range(len(keys))
width = 0.5

#select the next sub-plot
plot.subplot(211)


#for each result, compare it to the expected result
errors = [0]*len(keys)
correct = [0]*len(keys)
for i in xrange(len(data)):
    answer = int(solver.predict(data[i]))
    print answer
    if answer != outputs[i]:
        errors[answer] += 1
    else:
        correct[answer] += 1
    

#do the bar graphs
corr = plot.bar([i for i in xrange(len(keys))],correct,width,color='b')
err = plot.bar([i for i in xrange(len(keys))],errors,width,bottom=correct,color='r')

#label the x and y axes
plot.ylabel("# Classifications")
plot.xticks([i+width/2 for i in xrange(len(keys))],keys)

#do the legend
#plot.legend([corr,err],["Correct","Error"])

#show the plot
plot.show()
