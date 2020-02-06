
#ifndef __SOFTPLUS_H
#define __SOFTPLUS_H


class SoftPlus{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  SoftPlus(){
    n=0;
    p=0;
  }
  
  SoftPlus(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat SoftPlus::Get_Out(){
  return Out;
}

arma::mat SoftPlus::Get_dOut(){
  return dOut;
}

void SoftPlus::forward(arma::mat X){
  
  Out = log(1+exp(X)) ;
  

}


void SoftPlus::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut/(1+exp(- _X));

}


#endif



















