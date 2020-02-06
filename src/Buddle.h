
#ifndef __BUDDLE_H
#define __BUDDLE_H


class Buddle{
  
protected:
  
  
  double loss;
  double Accuracy;
  
  int p;
  int r;
  int n;
  int nHiddenLayer;
  double d_init_weight;
  double d_learning_rate;
  
  /////////////////
  
  int bBatch; 
  int bDrop;
  int bTest;
  double drop_ratio;
  
  String strOpt;
  String strType;
  
  
  int bRand;
  String strDist;
  
  
  
  //////////////////////
  
  Layer* Arr_Layer;
  
  SoftmaxLoss sml;
  L2loss l2loss;
  
  arma::vec HiddenLayer;
  arma::mat bOut;
  arma::mat mOut;
  arma::mat mEntropy;

  arma::mat dOut;
  arma::mat Final_dOut;
  
  
  
public:
  
  Buddle(){
    p=1;
    r=1;
    n=1;
  }
  
  Buddle(int _p, int _r, int _n, int _nHiddenLayer, arma::vec _HiddenLayer, String _strType, String* strVec, String _strOpt,
      double _d_learning_rate, double _d_init_weight, int _bBatch, int _bDrop, int _bTest, double _drop_ratio, 
      int _bRand, String _strDist) // Constructor
    : sml(SoftmaxLoss(_r, _n)), l2loss(L2loss(_r, _n)), HiddenLayer(_nHiddenLayer), 
      bOut(_r,_n), mOut(_r, _n), mEntropy(_n,1), dOut(_r,_n), Final_dOut(_p,_n) { // Default matrix member variable initialization
    
    
    loss = 0;
    Accuracy=0;
    
    bOut.zeros();
    mOut.zeros();
    mEntropy.zeros();
    
    p = _p;
    r = _r;
    n = _n;
    d_init_weight = _d_init_weight;
    d_learning_rate = _d_learning_rate;
    
    
    strType = _strType;
    strOpt = _strOpt;
    
    
    bRand=_bRand;
    strDist=_strDist;
    
    
    /////////////////
    bBatch = _bBatch; 
    bDrop = _bDrop;
    bTest = _bTest;
    drop_ratio = _drop_ratio;
    //////////////////////
    
    nHiddenLayer = _nHiddenLayer;
    HiddenLayer = _HiddenLayer;
    
    //Here!!!!
    
    // 
    
    
    String strAct("");
    int q1=0;
    int q2=0;
    int bAct=1;
    
    Arr_Layer = new Layer[nHiddenLayer+1];
    
    
    for(int i=1;i<=nHiddenLayer;i++){
      
      strAct = strVec[i-1];    //Activation function at each sublayer
      
      if(i==1){
        q1=p;
      }else{
        q1=q2;
      }
      
      q2 = HiddenLayer[i-1];
      
      Arr_Layer[i-1] = Layer(q2, q1, n, bAct, bBatch, bDrop, bTest, drop_ratio, 
                             d_learning_rate, d_init_weight, strAct, strOpt, bRand, strDist);
      
    }
    
    //Here!!!!
    ///// Last Affine Class
    bAct=0;
    q1 = q2;
    q2 = r;
    
    Arr_Layer[nHiddenLayer] = Layer(q2, q1, n, bAct, bBatch, bDrop, bTest, drop_ratio, 
                                    d_learning_rate, d_init_weight, strAct, strOpt , bRand, strDist);
    
  
  }
  
  ~Buddle(){
    delete []Arr_Layer;
  }
  
  
  Layer* Get_Arr_Layer();
  
  double Get_loss(){return loss;}
  double Get_Accuracy(){return Accuracy;}
  
  arma::mat Get_bOut(){return bOut;}
  arma::mat Get_mOut(){return mOut;}
  arma::mat Get_mEntropy(){return mEntropy;}
  arma::mat Get_dOut(){return dOut;}
  arma::mat Get_Final_dOut(){return Final_dOut;}
  
  void Set_loss();
  void Set_Accuracy(arma::mat t_Mat);
  
  void Set_Arr_Layer(Layer* _Arr_Layer);
  void forward(arma::mat X, arma::mat _t);
  void forward_predict(arma::mat X);
  
  void backward(arma::mat X, arma::mat _t);
  
};


void Buddle::Set_Arr_Layer(Layer* _Arr_Layer){
  
  for(int i=1;i<=(nHiddenLayer+1);i++){
    Arr_Layer[i-1].Set_W( _Arr_Layer[i-1].Get_W() )  ;
    Arr_Layer[i-1].Set_b( _Arr_Layer[i-1].Get_b() )  ;
  }
  
}



Layer* Buddle::Get_Arr_Layer(){
  return Arr_Layer;
}



void Buddle::forward(arma::mat X, arma::mat _t){
  
  
  for(int i=1;i<= (nHiddenLayer+1);i++){
    
    if(i==1){
      
      Arr_Layer[i-1].forward( X )  ;
      
    }else{
      Arr_Layer[i-1].forward( Arr_Layer[i-2].Get_Out() )  ;

    }
    
  }


  //Here!!!!
  /// Last Softmax
  bOut = Arr_Layer[nHiddenLayer].Get_Out() ;
  
  if(strType == strClassification){
    sml.forward(bOut, _t);
    mOut = sml.Get_y();
    mEntropy = sml.Get_Entropy();
    
  }else{
    l2loss.forward(bOut, _t);
    mOut = l2loss.Get_y();
    
  }
  
  
  
  //////////////////////////////////////////////////
}



void Buddle::forward_predict(arma::mat X){
  
  
  for(int i=1;i<= (nHiddenLayer+1);i++){
    
    if(i==1){
      
      Arr_Layer[i-1].forward( X )  ;
      
    }else{
      Arr_Layer[i-1].forward( Arr_Layer[i-2].Get_Out() )  ;
      
    }
    
  }
  
  
  //Here!!!!
  /// Last Softmax
  bOut = Arr_Layer[nHiddenLayer].Get_Out() ;
  
  if(strType == strClassification){
    sml.forward_predict(bOut);
    mOut = sml.Get_y();
    mEntropy = sml.Get_Entropy();
    
  }else{
    l2loss.forward_predict(bOut);
    mOut = l2loss.Get_y();
    
  }
  
  
  
  //////////////////////////////////////////////////
}






void Buddle::backward(arma::mat X, arma::mat _t){
  
  if(strType == strClassification){
    sml.backward(_t);
    dOut = sml.Get_dOut();
    
    
  }else{
    
    l2loss.backward(_t);
    dOut = l2loss.Get_dOut();
    
  }
  
  for(int i=(nHiddenLayer+1);i>=1;i--){
    if(i==(nHiddenLayer+1)){
      
      Arr_Layer[i-1].backward( Arr_Layer[i-2].Get_Out(), dOut) ;
      
    }else if(i==1){
      
      Arr_Layer[i-1].backward(X, Arr_Layer[i].Get_dOut()) ;
      
    }else{
      Arr_Layer[i-1].backward( Arr_Layer[i-2].Get_Out(), Arr_Layer[i].Get_dOut() ) ;
      
    }
    
    
  }
  
  
  Final_dOut = Arr_Layer[0].Get_dOut() ;
  
  
  
}


void Buddle::Set_loss(){
  

  if(strType == strClassification){
    loss = sml.Get_loss();
    
  }else{
    loss = l2loss.Get_loss();
  }

}


void Buddle::Set_Accuracy(arma::mat t_Mat){
  
  int nIndex=0;
  arma::vec x(r);
  double nAccuracy = 0;
  
  if(strType == strClassification){
    
    for(int i=1;i<=n;i++){
      x = mOut.col(i-1);
      nIndex = x.index_max();
      if( t_Mat(nIndex, i-1) == 1 ){
        nAccuracy++;
      }
      
    }
    
    Accuracy = nAccuracy/n;
  }else{
    Accuracy = -0.5* accu( (l2loss.Get_y() - t_Mat ) % (l2loss.Get_y() - t_Mat ))/n;
    
  }
  
}



#endif



















