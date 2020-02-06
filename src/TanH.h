
#ifndef __TANH_H
#define __TANH_H


class TanH{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  TanH(){
    n=0;
    p=0;
  }
  
  TanH(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _dOut);
  
  
};

arma::mat TanH::Get_Out(){
  return Out;
}

arma::mat TanH::Get_dOut(){
  return dOut;
}

void TanH::forward(arma::mat X){
  
  Out = (exp(X)-exp(-X))/(exp(X)+exp(-X));  
  

}


void TanH::backward(arma::mat _dOut){

  
  dOut =  (1 -  Out%Out ) % _dOut;


}


#endif



















