
#ifndef __L2LOSS_H
#define __L2LOSS_H

class L2loss{

private:
  
  int r;     //////  y = r x n matrix,  t = r x n matrix
  int n;     //////  loss = n x 1 matrix
  
  double loss;
  
  arma::mat y;  
  arma::mat dOut;   //// dOut = r x n matrix

public:
  
  L2loss(){
    r = 1;
    n = 1;
  }
  
  L2loss(int _r, int _n) // Constructor
    : y(_r, _n), dOut(_r,_n) { // Default matrix member variable initialization
    
    r = _r;
    n = _n;
    loss = 0;
    
    y.zeros();
    dOut.zeros();
  }
  
  arma::mat Get_y();
  arma::mat Get_dOut();
  double Get_loss();
  
  void forward(arma::mat X, arma::mat _t); 
  void forward_predict(arma::mat X); 
  void backward(arma::mat _t);
  
};



arma::mat L2loss::Get_y(){
  return y;
}


arma::mat L2loss::Get_dOut(){
  return dOut;
}

double L2loss::Get_loss(){
  
  return loss;
}


void L2loss::forward(arma::mat X, arma::mat _t){
  
  y = X;
  
  arma::mat yt = abs(y - _t);
  loss = 0.5* accu( yt%yt)/n;
  
}

void L2loss::forward_predict(arma::mat X){
  
  y = X;
  
}


void L2loss::backward(arma::mat _t){

  dOut = (y - _t)/n;
  
}



#endif



















