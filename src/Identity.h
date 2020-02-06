
#ifndef __IDENTITY_H
#define __IDENTITY_H


class Identity{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  Identity(){
    n=0;
    p=0;
  }
  
  Identity(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _dOut);
  
  
};

arma::mat Identity::Get_Out(){
  return Out;
}

arma::mat Identity::Get_dOut(){
  return dOut;
}

void Identity::forward(arma::mat X){
  
  Out = X;  
  

}


void Identity::backward(arma::mat _dOut){
  dOut = _dOut;

}


#endif



















