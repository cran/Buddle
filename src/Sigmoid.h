
#ifndef __SIGMOID_H
#define __SIGMOID_H


class Sigmoid{
  
private:
  
  int p;
  int n;
  
  arma::mat Out;
  arma::mat dOut;
   
public:
  
  Sigmoid(){
    n=0;
    p=0;
  }
  
  Sigmoid(int _p, int _n) // Constructor
    : Out(_p, _n), dOut(_p, _n) { // Default matrix member variable initialization
    
    n = _n;
    p = _p;
    
  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  void forward(arma::mat _X);
  void backward(arma::mat _dOut);
  
  
};

arma::mat Sigmoid::Get_Out(){
  return Out;
}

arma::mat Sigmoid::Get_dOut(){
  return dOut;
}

void Sigmoid::forward(arma::mat X){
  
  Out = 1/(1+exp(-X));  
  
  // arma::vec o1(p);
  // o1.ones();
  // 
  // arma::vec x(p);
  // x.zeros();
  // 
  // 
  // for(int i=1;i<=n;i++){
  //   x = X.col(i-1);
  //   Out.col(i-1) = o1/( 1+exp(-x));
  // }

}


void Sigmoid::backward(arma::mat _dOut){

  
  dOut =  ( (1-Out)%Out ) %_dOut;
  // arma::vec x(p);
  // x.zeros();
  // 
  // 
  // 
  // for(int i=1;i<=n;i++){
  //   x = Out.col(i-1);
  //   dOut.col(i-1) =  ( (1-x)%x )  % _dOut.col(i-1) ;
  // }


}


#endif



















