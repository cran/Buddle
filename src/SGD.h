
#ifndef __SGD_H
#define __SGD_H


class SGD{
  
private:
  
  double d_learning_rate;
  
  
public:
  
  SGD(){
    d_learning_rate=0;
  }
  
  SGD(double _d_learning_rate) { // Default matrix member variable initialization
    
    d_learning_rate = _d_learning_rate;
    
  }

  arma::mat Update(arma::mat W, arma::mat dW);
  
};


arma::mat SGD::Update(arma::mat W, arma::mat dW){
  
  arma::mat Out = W - d_learning_rate*dW;

  return Out;
}



#endif



















