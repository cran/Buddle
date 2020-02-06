
#ifndef __GBUDDLE_H
#define __GBUDDLE_H

class gBuddle: public Buddle {

private:
  
  String strLink;
  gLayer* Arr_gLayer;
  
  
public:
  
  gBuddle(int _p, int _r, int _n, int _nHiddenLayer, arma::vec _HiddenLayer, String _strType, String* strVec, String _strOpt,
          double _d_learning_rate, double _d_init_weight, int _bBatch, int _bDrop, int _bTest, double _drop_ratio, 
          int _bRand, String _strDist, String _strLink) : Buddle(_p, _r, _n, _nHiddenLayer, _HiddenLayer, _strType, strVec, 
          _strOpt, _d_learning_rate, _d_init_weight, _bBatch, _bDrop, _bTest,  _drop_ratio, _bRand, _strDist){
    
    strLink = _strLink;
    
    
    String strAct("");
    int q1=0;
    int q2=0;
    int bAct=1;
    
    Arr_gLayer = new gLayer[nHiddenLayer+1];
    
    
    for(int i=1;i<=nHiddenLayer;i++){
      
      strAct = strVec[i-1];    //Activation function at each sublayer
      
      if(i==1){
        q1=p;
      }else{
        q1=q2;
      }
      
      q2 = HiddenLayer[i-1];
      
      Arr_gLayer[i-1] = gLayer(q2, q1, n, bAct, bBatch, bDrop, bTest, drop_ratio, 
                             d_learning_rate, d_init_weight, strAct, strOpt, bRand, strDist, strLink);
      
    }
    
    //Here!!!!
    ///// Last Affine Class
    bAct=0;
    q1 = q2;
    q2 = r;
    
    Arr_gLayer[nHiddenLayer] = gLayer(q2, q1, n, bAct, bBatch, bDrop, bTest, drop_ratio, 
                                    d_learning_rate, d_init_weight, strAct, strOpt , bRand, strDist, strLink);
    
    
    
  }
  
  gLayer* Get_Arr_gLayer();
  void Set_Arr_gLayer(gLayer* _Arr_gLayer);
  void gforward(arma::mat X, arma::mat _t);
  void gforward_predict(arma::mat X);
  
  void gbackward(arma::mat X, arma::mat _t);
  
  
  
  ~gBuddle(){
    delete []Arr_gLayer;
  }
  
};


void gBuddle::Set_Arr_gLayer(gLayer* _Arr_gLayer){
  
  for(int i=1;i<=(nHiddenLayer+1);i++){
    Arr_gLayer[i-1].Set_W( _Arr_gLayer[i-1].Get_W() )  ;
    Arr_gLayer[i-1].Set_b( _Arr_gLayer[i-1].Get_b() )  ;
  }
  
}



gLayer* gBuddle::Get_Arr_gLayer(){
  return Arr_gLayer;
}



void gBuddle::gforward(arma::mat X, arma::mat _t){
  
  
  for(int i=1;i<= (nHiddenLayer+1);i++){
    
    if(i==1){
      
      Arr_gLayer[i-1].gforward( X )  ;
      
    }else{
      Arr_gLayer[i-1].gforward( Arr_gLayer[i-2].Get_Out() )  ;
      
    }
    
  }
  
  
  //Here!!!!
  /// Last Softmax
  bOut = Arr_gLayer[nHiddenLayer].Get_Out() ;
  
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



void gBuddle::gforward_predict(arma::mat X){
  
  
  for(int i=1;i<= (nHiddenLayer+1);i++){
    
    if(i==1){
      
      Arr_gLayer[i-1].gforward( X )  ;
      
    }else{
      Arr_gLayer[i-1].gforward( Arr_gLayer[i-2].Get_Out() )  ;
      
    }
    
  }
  
  
  //Here!!!!
  /// Last Softmax
  bOut = Arr_gLayer[nHiddenLayer].Get_Out() ;
  
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


void gBuddle::gbackward(arma::mat X, arma::mat _t){
  
  if(strType == strClassification){
    sml.backward(_t);
    dOut = sml.Get_dOut();
    
  }else{
    
    l2loss.backward(_t);
    dOut = l2loss.Get_dOut();
    
  }
  
  for(int i=(nHiddenLayer+1);i>=1;i--){
    if(i==(nHiddenLayer+1)){
      
      Arr_gLayer[i-1].gbackward( Arr_gLayer[i-2].Get_Out(), dOut) ;
      
    }else if(i==1){
      
      Arr_gLayer[i-1].gbackward(X, Arr_gLayer[i].Get_dOut()) ;
      
    }else{
      Arr_gLayer[i-1].gbackward( Arr_gLayer[i-2].Get_Out(), Arr_gLayer[i].Get_dOut() ) ;
      
    }
    
  }
  
  
  Final_dOut = Arr_gLayer[0].Get_dOut() ;
  
}

#endif



















