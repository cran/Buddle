#' Building a multi-layer feed-forward neural network model for statistical classification
#'
#'@param Data - Input matrix.
#'@param Label - Vector of training labels.
#'@param Train_Size - Size of data which is used for training .
#'@param Batch_Size - Batch size.
#'@param Optimization  - Method used to minimize loss fucntion. It can take one of "SGD", "Moment", or "AdaGrad."
#'@param Learning_Rate - Default is 0.05.
#'@param Iteration     - Number of iterations. Default is 100.
#'@param Layer         - Number of layers. Default is 3.
#'@param Neuron        - Number of neurons. Default is 20.
#'@param Activation    - The name of activation function. It takes either "Relu" or "Sigmoid." 
#'
#'@return Loss         - Vector of values of the loss function. 
#'@return W            - Matrix of weights in the first layer
#'@return b            - Vector of weights in the first layer
#'@return ZList        - List of matrices of weights in the middle layers
#'@return cList        - List of vectors of weights in the middle layers
#'@return ZFinal       - Matrix of weights in the final layer
#'@return cFinal       - Vector of weights in the final layer
#'@return Train_acc    - Accuracy of the classifier when applied to the train data
#'@return Test_acc     - Accuracy of the classifier when applied to the test data
#'@return Epoch        - Number of epoch. 

#'@examples
#'####################
#'n <- 50
#'p <- 3
#'Data <- matrix(runif(n*p, 0,50), nrow=n, ncol=p)  #### Generate n-by-p input matrix for data

#'Label = sample.int(n, n, replace=TRUE)            #### Generate n-by-1 vector for the label
#'Layer = 6                                      #### Number of layers
#'Neuron = 20                                    #### Number of neurons
#'lr = 0.005                                     #### Learning rate 
#'Iter = 100                                #### Iteration
#'Opt = "SGD"                              #### Method to optimize the loss function
#'Act = "Sigmoid"                          ##### Activation function
#'TrSize = 20                              ##### Train_Size
#'BatSize = 5                               ##### Batch_Size
#'DLResult = Buddle_Main(Data, Label, TrSize, BatSize, Opt, lr, Iter, Layer, Neuron, Act)
#'Loss_Vector = DLResult$Loss
#'Train_Accuracy = DLResult$Train_acc
#'Test_Accuracy = DLResult$Test_acc

#'@references
#'[1] Geron, A. Hand-On Machine Learning with Scikit-Learn and TensorFlow. Sebastopol: O'Reilly, 2017. Print.
#'@references
#'[2] Han, J., Pei, J, Kamber, M. Data Mining: Concepts and Techniques. New York: Elsevier, 2011. Print.  
#'@export
#'@seealso Buddle_Predict
#'@importFrom Rcpp evalCpp
#'@useDynLib Buddle


Buddle_Main = function(Data, Label, Train_Size, Batch_Size, Optimization = "SGD", Learning_Rate = 0.05, Iteration=100, Layer=3, Neuron = 20, Activation = "Sigmoid"){
  
  X = Data
  Y = Label
  
  dimm = dim(X)
  n=dimm[1]
  p = dimm[2]
  
  r = length(Y)+1
  
  
  step=Iteration
  nN = Train_Size
  nBatch = Batch_Size
  lr = Learning_Rate
  nLayer = Layer        # number of layers
  q = Neuron            # number of neurons
  
  if(Activation=="Sigmoid"){
    nModel = 0
  }else if(Activation=="Relu"){
    nModel = 1
  }
  
  if(Layer<3){
    Layer=3
  }
  
  if(Optimization=="SGD"){
    nMethod=0
  }else if(Optimization=="Momentum"){
    nMethod=1
  }else if(Optimization=="AdaGrad"){
    nMethod=2
  }
  
  
  
  
  XTrain = X[1:nN,]
  YTrain = Y[1:nN]
  
  tMat = matrix(0, r, nN)
  for(iter in 1:nN){
    t = rep(0, times=r)
    t[Y[iter]+1] = 1
    tMat[,iter] = t
  }
  
  XTest = X[(nN+1):n, ]
  YTest = Y[(nN+1):n]
  
  
  lst = MainFunc_Mat(YTest, XTest, YTrain, XTrain, tMat, nN, nBatch, nMethod, lr, step, nLayer, q, r, nModel)
  
  FinalList = list(Loss = lst[[1]], W = lst[[2]], b =lst[[3]], ZList=lst[[4]], cList = lst[[5]], 
                   ZFinal = lst[[6]], cFinal = lst[[7]], Train_acc = lst[[8]], Test_acc=lst[[9]], 
                   Epoch = lst[[10]], r=lst[[11]], Layer=lst[[12]], Neuron = lst[[13]], Activation=lst[[14]])
  
  
  return(FinalList)
}




#' Making a prediction based on the traind network model obtained from Buddle_Main()
#'
#'@param lst  - Feed-forward neural network model
#'@param xVec - Vector of data
#'
#'@return Predicted classification for given data. 

#'####################
#'@examples
#'n = 50
#'p = 3
#'Data = matrix(runif(n*p, 0,50), nrow=n, ncol=p)  #### Generate n-by-p input matrix for data
#'Label = sample.int(n, n, replace=TRUE)            #### Generate n-by-1 vector for the label
#'DLResult = Buddle_Main(Data, Label, 20, 5, "SGD", 0.01, 100, 6, 20, "Sigmoid")
#'
#'xVec=rep(0, times=p)
#'Index = Buddle_Predict(DLResult, xVec)        ###### Predict for given xVec
#'
#'@references
#'[1] Geron, A. Hand-On Machine Learning with Scikit-Learn and TensorFlow. Sebastopol: O'Reilly, 2017. Print.
#'@references
#'[2] Han, J., Pei, J, Kamber, M. Data Mining: Concepts and Techniques. New York: Elsevier, 2011. Print.  
#'@export
#'@seealso Buddle_Main


Buddle_Predict = function(lst, xVec){
  
  Loss_Val = lst[[1]];
  W = lst[[2]];
  b = lst[[3]];
  ZList = lst[[4]];
  cList = lst[[5]];
  ZFinal = lst[[6]];
  cFinal = lst[[7]];
  Train_acc = lst[[8]];
  Test_acc = lst[[9]];
  nEpochIter = lst[[10]];
  r = lst[[11]];
  nLayer = lst[[12]];
  q = lst[[13]];
  nModel = lst[[14]];
  
  y = rep(0, times=r)
  
  y = yVec(xVec, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
  max_yVal = max(y);
  
  nMaxIndex =  FindMaxIndex(y, max_yVal);
  
  return(nMaxIndex)
  
  
  
}