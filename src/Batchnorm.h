
#ifndef __BATCHNORM_H
#define __BATCHNORM_H


class Batchnorm{
  
private:
  
  int n;
  int p;
  
  arma::mat Original_X;
  arma::mat Out;
  arma::mat dOut;
  
public:
  
  Batchnorm(){
    n=0;
    p=0;
  
  }
  
  Batchnorm(int _p, int _n) // Constructor
    : Original_X(_p, _n), Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;

    Original_X.zeros();
    Out.zeros();
    dOut.zeros();
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat X);
  void backward(arma::mat _dOut);
  
  
};



arma::mat Batchnorm::Get_Out(){
  return Out;
}

arma::mat Batchnorm::Get_dOut(){
  return dOut;
}

void Batchnorm::forward(arma::mat X){
  
  Original_X = X;
  arma::vec x(p);

  for(int i = 1;i<=n; i++){
    x = X.col(i-1);
    Out.col(i-1) = BatchNorm(x);
  }
  
}


void Batchnorm::backward(arma::mat _dOut){
  
  arma::vec xi;
  arma::vec di;
  
  for(int i = 1;i<=n; i++){
    xi = Original_X.col(i-1);
    di = _dOut.col(i-1);
    dOut.col(i-1) = BackwardBatchNorm(xi, di);
  }
  
  
}

#endif



















