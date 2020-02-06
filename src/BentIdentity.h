
#ifndef __BENTIDENTITY_H
#define __BENTIDENTITY_H


class BentIdentity{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  BentIdentity(){
    n=0;
    p=0;
  }
  
  BentIdentity(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat BentIdentity::Get_Out(){
  return Out;
}

arma::mat BentIdentity::Get_dOut(){
  return dOut;
}

void BentIdentity::forward(arma::mat X){
  
  Out = ( X + ( sqrt(X%X +1 ) -1 )/2  ) ;  
  

}


void BentIdentity::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut% (1+ 0.5*_X /sqrt(1+_X%_X) ) ;

}


#endif



















