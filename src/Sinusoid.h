
#ifndef __SINUSOID_H
#define __SINUSOID_H


class Sinusoid{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  Sinusoid(){
    n=0;
    p=0;
  }
  
  Sinusoid(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat Sinusoid::Get_Out(){
  return Out;
}

arma::mat Sinusoid::Get_dOut(){
  return dOut;
}

void Sinusoid::forward(arma::mat X){
  
  Out = sin(X);  
  

}


void Sinusoid::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut% cos(_X) ;

}


#endif



















