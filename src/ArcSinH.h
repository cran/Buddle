
#ifndef __ARCSINH_H
#define __ARCSINH_H


class ArcSinH{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  ArcSinH(){
    n=0;
    p=0;
  }
  
  ArcSinH(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat ArcSinH::Get_Out(){
  return Out;
}

arma::mat ArcSinH::Get_dOut(){
  return dOut;
}

void ArcSinH::forward(arma::mat X){
  
  Out = log( X + sqrt( 1+X%X )  );  
  

}


void ArcSinH::backward(arma::mat _X, arma::mat _dOut){
  dOut = _dOut/sqrt(1+_X%_X) ;

}


#endif



















