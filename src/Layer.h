
#ifndef __LAYER_H
#define __LAYER_H


class Layer{
  
protected:  
  
  int q;
  int p;
  int n;
  int bAct;
  int bBatch;
  int bDrop;
  int bTest;
  double drop_ratio;
  double d_learning_rate;
  double d_initial_weight;
  
  int bRand;
  String strDist;
  
  
  String strAct;          ///  Activation function
  String strOpt;          ///   Optimization method 
  
  Affine affine;
  Relu relu;
  Sigmoid sigmoid;
  LeakyRelu leakyrelu;
  TanH tanh;
  ArcTan arctan;
  ArcSinH arcsinh;
  ElliotSig elliotsig;
  SoftPlus softplus;
  BentIdentity bentidentity;
  Sinusoid sinusoid;
  Identity identity;
  Gaussian gaussian;
  Sinc sinc;
  
  
  Dropout dropout;
  Batchnorm batchnorm;
  
  arma::mat W;
  arma::mat b;
  arma::mat dW;
  arma::mat db;
  
  arma::mat v;
  arma::mat dv;
  
  Optimization opt;
  arma::mat Out;
  arma::mat dOut;
  

public:
  
  Layer(){
    q=1;
    p=1;
    n=1;
  }
  
  Layer(int _q, int _p, int _n, int _bAct, int _bBatch, int _bDrop, int _bTest, double _drop_ratio, 
        double _d_learning_rate, double _d_initial_weight, String _strAct, String _strOpt, int _bRand, String _strDist)
    : affine(Affine(_q, _p, _n, _bRand, _strDist)), relu(Relu(_q,_n)), sigmoid(Sigmoid(_q,_n)), leakyrelu(LeakyRelu(_q,_n)),
      tanh(TanH(_q,_n)), arctan(ArcTan(_q,_n)), arcsinh(ArcSinH(_q,_n)),
      elliotsig(ElliotSig(_q,_n)), softplus(SoftPlus(_q,_n)),
      bentidentity(BentIdentity(_q,_n)), sinusoid(Sinusoid(_q,_n)), identity(Identity(_q,_n)),
      gaussian(Gaussian(_q,_n)), sinc(Sinc(_q,_n)),
      dropout( Dropout(_q, _n, _bTest, _drop_ratio)), batchnorm(Batchnorm(_q,_n)),
      W(_q,_p), b(_q,1), dW(_q,_p), db(_q,1), v((_q+2),1), dv((_q+2),1),
      opt(Optimization(_q, _p, _bRand, _d_learning_rate, _strOpt)),
      Out(_q,_n), dOut(_p,_n) { // Default matrix member variable initialization
    
    
    bRand=_bRand;
    strDist = _strDist;
    
    q = _q;
    p = _p;
    n = _n;
    
    
    bAct = _bAct;
    bBatch=_bBatch;
    bDrop=_bDrop;
    bTest=_bTest;
    drop_ratio=_drop_ratio;
    d_learning_rate = _d_learning_rate;
    d_initial_weight = _d_initial_weight;
    
    strOpt = _strOpt;
    strAct = _strAct; 
     
    W.randn(q, p);
    b.zeros();
    
    v.randu((q+2),1);
    v(1,0) += 1e-5;
    
    W *= d_initial_weight;
    b *= d_initial_weight;
    
    if(strAct == strRelu || strAct == strLeakyRelu){
      W /= sqrt(p/2);
    }else{
      W /= sqrt(p);
    }
    
    
    
  }
  
  
  arma::mat Get_Out();
  arma::mat Get_dOut();  
  
  arma::mat Get_W();
  arma::mat Get_b();  
  
  arma::mat Get_dW();
  arma::mat Get_db();  
  
  arma::mat Get_v();
  arma::mat Get_dv();  
  
  
  void Set_W(arma::mat _W);
  void Set_b(arma::mat _b);
  
  void Set_v(arma::mat _v);
  
  
  void forward(arma::mat X);
  void backward(arma::mat X, arma::mat dOut);
  
  
};

arma::mat Layer::Get_Out(){
  return Out;
}

arma::mat Layer::Get_dOut(){
  return dOut;
}

arma::mat Layer::Get_W(){
  
  return W;
}


arma::mat Layer::Get_b(){
  return b;
}

arma::mat Layer::Get_dW(){
  return dW;
}

arma::mat Layer::Get_db(){
  return db;
}


arma::mat Layer::Get_v(){
  return v;
}

arma::mat Layer::Get_dv(){
  return dv;
}




void Layer::Set_W(arma::mat _W){
  
  W = _W;
}


void Layer::Set_b(arma::mat _b){
  
  b=_b;
}

void Layer::Set_v(arma::mat _v){
  
  v=_v;
}


void Layer::forward(arma::mat X){
  

  affine.Set_W(W);
  affine.Set_b(b);
  affine.Set_v(v);
  
  affine.forward(X);
  
  Out = affine.Get_Out();
  
  if(bAct == 1){
    
    if(bBatch==1){
      batchnorm.forward(Out);
      Out = batchnorm.Get_Out();
    }
    
    if(strAct == strRelu){
      relu.forward(Out);
      Out = relu.Get_Out();
    }else if(strAct == strSigmoid){
      sigmoid.forward(Out);
      Out = sigmoid.Get_Out();
    }else if(strAct == strLeakyRelu){
      leakyrelu.forward(Out);
      Out = leakyrelu.Get_Out();
    }else if(strAct == strTanH){
      tanh.forward(Out);
      Out = tanh.Get_Out();
      
    }else if(strAct == strArcTan){
      arctan.forward(Out);
      Out = arctan.Get_Out();
      
    }else if(strAct == strArcSinH){
      arcsinh.forward(Out);
      Out = arcsinh.Get_Out();
      
    }else if(strAct == strElliotSig){
      elliotsig.forward(Out);
      Out = elliotsig.Get_Out();
      
    }else if(strAct == strSoftPlus){
      softplus.forward(Out);
      Out = softplus.Get_Out();
      
    }else if(strAct == strBentIdentity){
      bentidentity.forward(Out);
      Out = bentidentity.Get_Out();
      
    }else if(strAct == strSinusoid){
      sinusoid.forward(Out);
      Out = sinusoid.Get_Out();
      
    }else if(strAct == strGaussian){
      gaussian.forward(Out);
      Out = gaussian.Get_Out();
      
    }else if(strAct == strSinc){
      sinc.forward(Out);
      Out = sinc.Get_Out();
      
    }else{
      identity.forward(Out);
      Out = identity.Get_Out();
    }
    
    if(bDrop==1){
      dropout.forward(Out);
      Out = dropout.Get_Out();
    }
    
    
  }
  
}


void Layer::backward(arma::mat _X, arma::mat _dOut){

  if(bAct == 1){
    if(bDrop==1){
      dropout.backward(_dOut);
      _dOut = dropout.Get_dOut();
    }
    
    if(strAct == strRelu){
      relu.backward(_dOut);
      _dOut = relu.Get_dOut();
    }else if(strAct == strSigmoid){
      sigmoid.backward(_dOut);
      _dOut = sigmoid.Get_dOut();
    }else if(strAct == strLeakyRelu){
      leakyrelu.backward(_dOut);
      _dOut = leakyrelu.Get_dOut();
    }else if(strAct == strTanH){
      tanh.backward(_dOut);
      _dOut = tanh.Get_dOut();
    }else if(strAct == strArcTan){
      
      if(bDrop==1){
        arctan.backward(dropout.Get_Out(), _dOut);
      }else{
        arctan.backward(affine.Get_Out(), _dOut);
      }
        _dOut = arctan.Get_dOut();
    }else if(strAct == strArcSinH){
      
      if(bDrop==1){
        arcsinh.backward(dropout.Get_Out(), _dOut);
      }else{
        arcsinh.backward(affine.Get_Out(), _dOut);
      }
      _dOut = arcsinh.Get_dOut();
    }else if(strAct == strElliotSig){
      
      if(bDrop==1){
        elliotsig.backward(dropout.Get_Out(), _dOut);
      }else{
        elliotsig.backward(affine.Get_Out(), _dOut);
      }
      _dOut = elliotsig.Get_dOut();
    }else if(strAct == strSoftPlus){
      
      if(bDrop==1){
        softplus.backward(dropout.Get_Out(), _dOut);
      }else{
        softplus.backward(affine.Get_Out(), _dOut);
      }
      _dOut = softplus.Get_dOut();
    }else if(strAct == strBentIdentity){
      
      if(bDrop==1){
        bentidentity.backward(dropout.Get_Out(), _dOut);
      }else{
        bentidentity.backward(affine.Get_Out(), _dOut);
      }
      _dOut = bentidentity.Get_dOut();
    }else if(strAct == strSinusoid){
      
      if(bDrop==1){
        sinusoid.backward(dropout.Get_Out(), _dOut);
      }else{
        sinusoid.backward(affine.Get_Out(), _dOut);
      }
      _dOut = sinusoid.Get_dOut();
    }else if(strAct == strGaussian){
      
      if(bDrop==1){
        gaussian.backward(dropout.Get_Out(), _dOut);
      }else{
        gaussian.backward(affine.Get_Out(), _dOut);
      }
      _dOut = gaussian.Get_dOut();
    }else if(strAct == strSinc){
      
      if(bDrop==1){
        sinc.backward(dropout.Get_Out(), _dOut);
      }else{
        sinc.backward(affine.Get_Out(), _dOut);
      }
      _dOut = sinc.Get_dOut();
    }else{
      identity.backward(_dOut);
      _dOut = identity.Get_dOut();
      
    }
    
    
    if(bBatch==1){
      batchnorm.backward(_dOut);
      _dOut = batchnorm.Get_dOut();
    }
    
  }
  
  affine.backward(_X, _dOut);
  dOut = affine.Get_dOut();
  
  dW = affine.Get_dW();
  db = affine.Get_db();
  dv = affine.Get_dv();  
  ////////////// Optimization
  
  opt.Set_W(W);
  opt.Set_b(b);
  opt.Set_dW(dW);
  opt.Set_db(db);
  opt.Update();
  
  W = opt.Get_W();
  b = opt.Get_b();
  
  
}





#endif



















