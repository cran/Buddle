
#ifndef __DROPOUT_H
#define __DROPOUT_H


class Dropout{
  
private:
  
  int n;
  int p;
  int bTest;
  double drop_ratio;
  
  arma::mat Mask;
  arma::mat Out;
  arma::mat dOut;
  
public:
  
  Dropout(){
    n=0;
    p=0;
    bTest=0;
    drop_ratio = 0;
  }
  
  Dropout(int _p, int _n, int _bTest, double _drop_ratio) // Constructor
    : Mask(_p, _n), Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    bTest = _bTest;
    drop_ratio = _drop_ratio;
    
    Mask.zeros();
    Out.zeros();
    dOut.zeros();
    
  }
  
  arma::mat Get_Mask();
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat X);
  void backward(arma::mat _dOut);
  
  
};

arma::mat Dropout::Get_Mask(){
  return Mask;
}


arma::mat Dropout::Get_Out(){
  return Out;
}

arma::mat Dropout::Get_dOut(){
  return dOut;
}

void Dropout::forward(arma::mat X){
  
  arma::mat tmpMat(p,n);
  tmpMat.randu();
  
  if(bTest==0){
    Mask = Masking(tmpMat, drop_ratio);
    Out = X % Mask;
  }else{
    Out = (1-drop_ratio)*X; 
  }
  
}


void Dropout::backward(arma::mat _dOut){
  
  dOut = Mask % _dOut;  
  
}

#endif



















