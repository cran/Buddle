
#ifndef __ELLIOTSIG_H
#define __ELLIOTSIG_H


class ElliotSig{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  ElliotSig(){
    n=0;
    p=0;
  }
  
  ElliotSig(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat ElliotSig::Get_Out(){
  return Out;
}

arma::mat ElliotSig::Get_dOut(){
  return dOut;
}

void ElliotSig::forward(arma::mat X){
  
  Out = X / (1+abs(X));
  

}


void ElliotSig::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut/  ((1+ abs(_X)) % (1+ abs(_X))) ;

}


#endif



















