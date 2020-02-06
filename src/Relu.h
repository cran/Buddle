
#ifndef __RELU_H
#define __RELU_H


class Relu{
  
private:
  
  int n;
  int p;
  
  arma::mat Out;
  arma::mat dOut;
  
public:
  
  Relu(){
    n=0;
    p=0;
  }
  
  Relu(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;

    Out.zeros();
    dOut.zeros();
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat X);
  void backward(arma::mat _dOut);
  
  
};

arma::mat Relu::Get_Out(){
  return Out;
}

arma::mat Relu::Get_dOut(){
  return dOut;
}

void Relu::forward(arma::mat X){
  double cut_off=0;
  Out = Masking(X, cut_off) % X;

}


void Relu::backward(arma::mat _dOut){
  
  dOut = Out % _dOut;  
  
  // arma::vec x(p);
  // x.zeros();
  // 
  // for(int i=1;i<=n;i++){
  //   x = Out.col(i-1);
  //   dOut.col(i-1) =  x  % _dOut.col(i-1) ;
  // }
}

#endif



















