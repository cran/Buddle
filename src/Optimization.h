
#ifndef __OPTIMIZATION_H
#define __OPTIMIZATION_H

class Optimization{

private:
  
  int q;
  int p;
  int bRand;
  
  double d_learning_rate;
  double momentum_alpha;
  double d_decay;
  int nIter;
  double d_lr_t;
  double beta1;
  double beta2;
  
  String strOpt;
  
  arma::mat W;
  arma::mat b;
  arma::mat dW;
  arma::mat db;
  
  arma::mat v;
  arma::mat dv;
  
  
  arma::mat vW;
  arma::mat vb;
  arma::mat vv;
  
  arma::mat hW;
  arma::mat hb;
  arma::mat hv;
  
  arma::mat mW;
  arma::mat mb;
  arma::mat mv;
  
  arma::mat nW;
  arma::mat nb;
  arma::mat nv;
  
public:
  
  Optimization(){
    q = 1;
    p = 1;
  }
  
  Optimization(int _q, int _p, int _bRand, double _d_learning_rate, String _strOpt) // Constructor
    : W(_q,_p), b(_q,1), dW(_q,_p), db(_q,1), v((_q+2),1), dv((_q+2), 1),  
      vW(_q,_p), vb(_q,1), vv((_q+2),1), 
      hW(_q,_p), hb(_q,1), hv((_q+2),1), 
      mW(_q,_p), mb(_q,1), mv((_q+2),1), 
      nW(_q,_p), nb(_q,1), nv((_q+2),1)  { // Default matrix member variable initialization
    
    q = _q;
    p = _p;
    
    bRand = _bRand;
    
    d_learning_rate=_d_learning_rate;
    momentum_alpha = 0.9;
    d_decay = 0.99;
    nIter=0;
    d_lr_t = d_learning_rate;
    beta1=0.9;
    beta2=0.999;
    
    strOpt = _strOpt;
    
    W.zeros();
    b.zeros();
    dW.zeros();
    db.zeros();
    
    v.zeros();
    dv.zeros();
    
    vW.zeros();
    vb.zeros();
    hW.zeros();
    hb.zeros();
    
    mW.zeros();
    mb.zeros();
    nW.zeros();
    nb.zeros();

  }
  
  arma::mat Get_W();
  arma::mat Get_b();
  arma::mat Get_v();
  
  
  void Set_W(arma::mat _W);
  void Set_b(arma::mat _b);
  void Set_dW(arma::mat _dW);
  void Set_db(arma::mat _db);
  
  void Set_v(arma::mat _v);
  void Set_dv(arma::mat _dv);
  
  
  void Update();
  
};

arma::mat Optimization::Get_W(){
  return W;
}


arma::mat Optimization::Get_b(){
  return b;
}

arma::mat Optimization::Get_v(){
  return v;
}


void Optimization::Set_W(arma::mat _W){
  W = _W;
}

void Optimization::Set_b(arma::mat _b){
  b = _b;
}

void Optimization::Set_dW(arma::mat _dW){
  dW = _dW;
}

void Optimization::Set_db(arma::mat _db){
  db = _db;
}

void Optimization::Set_v(arma::mat _v){
  v = _v;
}

void Optimization::Set_dv(arma::mat _dv){
  dv = _dv;
}



void Optimization::Update(){
  
  double delta = 1e-5;
  double dtmp = 0;
  
  if(strOpt == strSGD){
    
    W -= d_learning_rate*dW;
    b -= d_learning_rate*db;
    
    if(bRand==1){
      v -= d_learning_rate*dv;
    }
    
    
  }else if(strOpt == strMomentum){
    
    vW = momentum_alpha*vW - d_learning_rate * dW;
    vb = momentum_alpha*vb - d_learning_rate * db;
    
    W += vW;
    b += vb;
    
    if(bRand==1){
      vv = momentum_alpha*vv - d_learning_rate * dv;
      v += vv;
    }
    
    
    
    
  }else if(strOpt == strNesterov){
    
    dtmp = momentum_alpha;
    
    vW = dtmp * vW - d_learning_rate*dW;
    vb = dtmp * vb - d_learning_rate*db;
    
    vW = (dtmp*dtmp)* vW - (1+dtmp)*d_learning_rate * dW;
    vb = (dtmp*dtmp)* vb - (1+dtmp)*d_learning_rate * db;
    
    W += vW;
    b += vb;
    
    if(bRand==1){
      vv = dtmp * vv - d_learning_rate*dv;
      vv = (dtmp*dtmp)* vv - (1+dtmp)*d_learning_rate * dv;
    }
    
    
    
  }else if(strOpt == strAdaGrad){
    
    
    hW += (dW%dW);
    hb += (db%db);
    
    W -= d_learning_rate*(dW/(sqrt(hW)+delta));
    b -= d_learning_rate*(db/(sqrt(hb)+delta));
    
    if(bRand==1){
      hv += (dv%dv);
      v -= d_learning_rate*(dv/(sqrt(hv)+delta));
    }
    
    
    
  }else if(strOpt == strRMSprop){
    
    hW *= d_decay;
    hb *= d_decay;
    
    hW += (1-d_decay)* (dW%dW);
    hb += (1-d_decay)* (db%db);
    
    W -= d_learning_rate*(dW/(sqrt(hW)+delta) );
    b -= d_learning_rate*(db/(sqrt(hb)+delta));
    
    if(bRand==1){
      hv *= d_decay;
      hv += (1-d_decay)* (dv%dv);
      v -= d_learning_rate*(dv/(sqrt(hv)+delta));
    }
    
  }else if(strOpt == strAdam){
    
    nIter++;
    d_lr_t = d_learning_rate*(1-beta2)/(1-beta1);
    
    mW += (1-beta1)*(dW-mW);
    mb += (1-beta1)*(db-mb);
    
    nW += (1-beta2)*(dW%dW - nW);
    nb += (1-beta2)*(db%db - nb);
    
    W -= d_lr_t*mW/sqrt(nW+delta);
    b -= d_lr_t*mb/sqrt(nb+delta);
    
    
    if(bRand==1){
      
      mv += (1-beta1)*(dv-mv);
      nv += (1-beta2)*(dv%dv - nv);
      v -= d_lr_t*mb/sqrt(nv+delta);
      
    }
    
    
  }else{
    
    W -= d_learning_rate*dW;
    b -= d_learning_rate*db;
    
    if(bRand==1){
      v -= d_learning_rate*dv;
    }
    
  }
  
  v(1,0) = abs(v(1,0))+delta;   ////// The second element of v is sigma, and hence should be >0
  
}






#endif



















