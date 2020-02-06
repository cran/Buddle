
#ifndef __SINC_H
#define __SINC_H


class Sinc{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  Sinc(){
    n=0;
    p=0;
  }
  
  Sinc(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat Sinc::Get_Out(){
  return Out;
}

arma::mat Sinc::Get_dOut(){
  return dOut;
}

void Sinc::forward(arma::mat X){
  
  double del = 1e-7;
  arma::uvec ind = find(X == 0);
  
  X.ones();
  Out = X%sin(X) / (X+del);
  Out.elem( ind ).ones();

}


void Sinc::backward(arma::mat X, arma::mat _dOut){
  
  double del = 1e-7;
  arma::uvec ind = find(X == 0);
  
  dOut = _dOut % ( cos(X)/(X+del) - sin(X)/( X%X+del ));
  dOut.elem( ind ).zeros();
  

}


#endif



















