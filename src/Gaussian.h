
#ifndef __GAUSSIAN_H
#define __GAUSSIAN_H


class Gaussian{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  Gaussian(){
    n=0;
    p=0;
  }
  
  Gaussian(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat Gaussian::Get_Out(){
  return Out;
}

arma::mat Gaussian::Get_dOut(){
  return dOut;
}

void Gaussian::forward(arma::mat X){
  Out = exp(-X%X);  
}


void Gaussian::backward(arma::mat _X, arma::mat _dOut){
  dOut = 2*_dOut % _X % exp(-_X%_X);

}


#endif



















