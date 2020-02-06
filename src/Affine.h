
#ifndef __AFFINE_H
#define __AFFINE_H


class Affine{
  
protected:
  
  int q;     //////  W = q x p matrix,  b = q x 1 vector
  int p;     //////  X = p x n matrix
  int n;     //////  WX + b = q x n matrix, Wx+b = q x 1 vector    
  
  arma::mat Out;   //// q x n
  arma::mat dOut;  ////  _dOut = the same dimension with Out, q x n
                   ////  dOut = the same dimension with X,  p x n
  
private:
  
  arma::mat dW;    //// the same dimension with W, q x p
  arma::mat db;    //// the same dimension with b, q x 1
  
  arma::mat W;
  arma::mat b;

  int bRand;
  FInv finv;
  String strDist;
  
  arma::mat v;
  arma::mat dv;
  
  
public:
  
  Affine(){
    q=1;
    p=1;
    n=1;
  }
  
  Affine(int _q, int _p, int _n, int _bRand, String _strDist) // Constructor
    : Out(_q, _n), dOut(_p, _n), dW(_q, _p), db(_q, 1),
      W(_q, _p), b(_q,1), finv(FInv(_q, 1, _strDist)), v((_q+2),1), dv((_q+2),1) { // Default matrix member variable initialization
    
    
    q = _q;
    p = _p;
    n = _n;
    
    bRand = _bRand;
    strDist = _strDist;
    
    Out.zeros();
    dOut.zeros();
    
    W.zeros();
    b.zeros();
    
    dW.zeros();
    db.zeros();
    
    v.zeros();
    dv.zeros();
    
  }
  
  
  
  arma::mat Get_Out();
  arma::mat Get_dOut();
  
  arma::mat Get_dW();
  arma::mat Get_db();
  
  arma::mat Get_dv();
  
  
  void Set_W(arma::mat _W);
  void Set_b(arma::mat _b);
  
  void Set_v(arma::mat _v);
  
  
  void forward(arma::mat _X);
  void backward(arma::mat _X, arma::mat _dOut);
  
  
};

arma::mat Affine::Get_Out(){
  return Out;
}

arma::mat Affine::Get_dOut(){
  return dOut;
}

arma::mat Affine::Get_dW(){
  return dW;
}

arma::mat Affine::Get_db(){
  return db;
}

arma::mat Affine::Get_dv(){
  return dv;
}


void Affine::Set_W(arma::mat _W){
  W = _W;
}

void Affine::Set_b(arma::mat _b){
  b = _b;
}

void Affine::Set_v(arma::mat _v){
  v = _v;
  
  v.rows(0,1) = _v.rows(0,1);
  
  arma::vec x(q);
  x.randu(q);
  
  for(int i=1;i<=q;i++){
    v(i-1+2,0) = x[i-1];
  }
  
}


void Affine::forward(arma::mat _X){
  
  arma::mat OneMat(1, n);
  OneMat.ones();

  Out = W*_X+ b*OneMat;
  
  if(bRand==1){
    
    finv.forward(v);
    Out += finv.Get_Out()*OneMat;
  }
  

}


void Affine::backward(arma::mat _X, arma::mat _dOut){

  dW = _dOut* _X.t();
  db = sum(_dOut, 1);
  dOut = W.t()* _dOut;
  
  if(bRand==1){
    
    finv.backward(v, db);   //// db or sum(_dOut, 1);
    dv = finv.Get_dOut();
    
  }
  
  

}

class gAffine: public Affine {
  
private:
  
  int q2;
  arma::mat V;
  arma::mat dV;
  
  arma::mat tmp_Out;
  arma::mat tmp_dOut;
  
  Link link;
  FInv finv2;
  
  String strLink;
  
public:
  
  gAffine(){
    q=1;
    p=1;
    n=1;
  }
  
  gAffine( int _q, int _p, int _n, int _bRand, String _strDist, String _strLink) // Constructor
    : Affine(_q, _p, _n, _bRand, _strDist), V((_q+2), _n), dV((_q+2), _n), 
      tmp_Out(_q,n), tmp_dOut( (_q+2),n), link(Link(_q, _n, _strLink)), finv2(FInv(_q, n, _strLink)){
      
      strLink = _strLink;
      
      q2 = _q+2;  
      V.zeros();
      dV.zeros();
      
      tmp_Out.zeros();
      tmp_dOut.zeros();
  }
  
  arma::mat Get_dV();
  void Set_V(arma::mat _V);
  
  void gforward(arma::mat _X);
  void gbackward(arma::mat _X, arma::mat _dOut);
  

};

void gAffine::Set_V(arma::mat _V){
  
  V.rows(0,1) = _V.rows(0,1);
  
  arma::mat tmpU(q,n);
  tmpU.randu(q,n);
  
  V.rows(3-1, q2-1) = tmpU;
  
}



void gAffine::gforward(arma::mat _X){
  
  forward(_X);            ///// Get Out  after affine layer
  tmp_Out = Out;          ////// Out:qxn                      
  link.forward(Out);
  V = link.Get_Out();     ///// Pass Out to link function to make V: q2xn
  finv2.forward(V);        ///// Pass Out to FInv node 
  Out = finv2.Get_Out();      ///// Out:qxn
  
  
}


void gAffine::gbackward(arma::mat _X, arma::mat _dOut){
                                 ////  _dOut :qxn 
  finv2.backward(V, _dOut);    ////   Get dOut after FInv
  tmp_dOut = finv2.Get_dOut();   //// tmp_dOut: q2xn
  link.backward(tmp_Out, tmp_dOut);
  
  _dOut = link.Get_dOut();          /////_dOut:qxn
  
  backward(_X, _dOut);        ///// Pass _dOut to affine layer 
                              /////  Get dW, db, dV
  
}



#endif



















