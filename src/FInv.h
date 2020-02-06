
#ifndef __FINV_H
#define __FINV_H


class FInv{
  
private:
  
  int q;     //////  v: (q+2)x1 or (q+1) vectpr
  int n;
  String strDist;
  
  arma::mat Out;   //// q x 1
  arma::mat dOut;  ////  _dOut = the same dimension with Out, q x n
                   ////  dOut = the same dimension with X,  p x n
  
public:
  
  FInv(){
    q=1;
    n=1;
  }
  
  FInv(int _q, int _n, String _strDist ) // Constructor
    : Out(_q, _n), dOut((_q+2), _n) { // Default matrix member variable initialization
    
    q = _q;
    n = _n;
  
    strDist = _strDist;

    Out.zeros();
    dOut.zeros();

  }
  
  arma::mat Get_Out();
  arma::mat Get_dOut();

  void forward(arma::mat v);
  void backward(arma::mat v, arma::mat _dOut);
  
  
};

arma::mat FInv::Get_Out(){
  return Out;
}

arma::mat FInv::Get_dOut(){
  return dOut;
}



void FInv::forward(arma::mat v){
  
  Out = fi(v, strDist);            ///// V q2xn, Out qxn 
  
}


void FInv::backward(arma::mat v, arma::mat _dOut){

  dOut = dfi(v, Out, _dOut, strDist);     ///_dOut: qxn
                                          ///dOut: q2xn  
}


#endif



















