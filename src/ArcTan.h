
#ifndef __ARCTAN_H
#define __ARCTAN_H


class ArcTan{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  ArcTan(){
    n=0;
    p=0;
  }
  
  ArcTan(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat ArcTan::Get_Out(){
  return Out;
}

arma::mat ArcTan::Get_dOut(){
  return dOut;
}

void ArcTan::forward(arma::mat X){
  
  Out = atan(X);  
  

}


void ArcTan::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut/(1+_X%_X) ;

}


#endif



















