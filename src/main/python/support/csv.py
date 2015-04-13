from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import one_to_n
from svm.svm import *

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
            out_data.append(samples[-1])
        else:
            out_data.append([float(i) for i in samples[-outputs:]])
        
    
    processed_out_data = out_data
    
    #process multiclass encoding
    if multiclass:
        processed_out_data = []
        #get all the unique values for classes
        keys = []
        for d in out_data:
            if d not in keys:
                keys.append(d)
        keys.sort()
        #encode all data
        for d in out_data:
            processed_out_data.append(one_to_n(keys.index(d),len(keys)))
    
    #create the dataset
    dataset = SupervisedDataSet(len(in_data[0]),len(processed_out_data[0]))
    for i in xrange(len(out_data)):
        dataset.addSample(in_data[i],processed_out_data[i])
    
    #return the keys if we have
    if multiclass:
        return dataset,keys # a multiclass classifier
    else:
        return dataset

def loadSVMProblem(filename,multiclass=True,outputs=1,separator=','):
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
            out_data.append(samples[-1])
        else:
            out_data.append([float(i) for i in samples[-outputs:]])
        
    #process multiclass encoding
    keys = []
    if multiclass:
        processed_out_data = []
        #get all the unique values for classes
        keys = []
        for d in out_data:
            if d not in keys:
                keys.append(d)
        keys.sort()
    
    #use libsvm's data container:
    return svm_problem([keys.index(i) for i in out_data],in_data),in_data,[keys.index(i) for i in out_data],keys