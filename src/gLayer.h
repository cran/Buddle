
#ifndef __GLAYER_H
#define __GLAYER_H


class gLayer : public Layer{
  
private:  
  
  String strLink;
  gAffine gaffine;

public:
  gLayer(){
    q=1;
    p=1;
    n=1;
  }
  
  gLayer(int _q, int _p, int _n, int _bAct, int _bBatch, int _bDrop, int _bTest, double _drop_ratio, 
        double _d_learning_rate, double _d_initial_weight, String _strAct, String _strOpt, 
        int _bRand, String _strDist, String _strLink) :Layer(_q, _p, _n, _bAct, _bBatch, _bDrop, _bTest, _drop_ratio, 
   _d_learning_rate, _d_initial_weight, _strAct, _strOpt, _bRand, _strDist), 
  gaffine(gAffine(_q, _p, _n, _bRand, _strDist, _strLink)) { // Default matrix member variable initialization
    
    strLink = _strLink;
    
  }
  
  
  void gforward(arma::mat X);
  void gbackward(arma::mat X, arma::mat dOut);
  
  
};



void gLayer::gforward(arma::mat X){
  
  gaffine.Set_W(W);
  gaffine.Set_b(b);
  
  if(bRand==1){
    gaffine.Set_v(v);
  }
  
  gaffine.gforward(X);
  Out = gaffine.Get_Out();
  
  
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


void gLayer::gbackward(arma::mat _X, arma::mat _dOut){

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
        arctan.backward(gaffine.Get_Out(), _dOut);
      }
        _dOut = arctan.Get_dOut();
    }else if(strAct == strArcSinH){
      
      if(bDrop==1){
        arcsinh.backward(dropout.Get_Out(), _dOut);
      }else{
        arcsinh.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = arcsinh.Get_dOut();
    }else if(strAct == strElliotSig){
      
      if(bDrop==1){
        elliotsig.backward(dropout.Get_Out(), _dOut);
      }else{
        elliotsig.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = elliotsig.Get_dOut();
    }else if(strAct == strSoftPlus){
      
      if(bDrop==1){
        softplus.backward(dropout.Get_Out(), _dOut);
      }else{
        softplus.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = softplus.Get_dOut();
    }else if(strAct == strBentIdentity){
      
      if(bDrop==1){
        bentidentity.backward(dropout.Get_Out(), _dOut);
      }else{
        bentidentity.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = bentidentity.Get_dOut();
    }else if(strAct == strSinusoid){
      
      if(bDrop==1){
        sinusoid.backward(dropout.Get_Out(), _dOut);
      }else{
        sinusoid.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = sinusoid.Get_dOut();
    }else if(strAct == strGaussian){
      
      if(bDrop==1){
        gaussian.backward(dropout.Get_Out(), _dOut);
      }else{
        gaussian.backward(gaffine.Get_Out(), _dOut);
      }
      _dOut = gaussian.Get_dOut();
    }else if(strAct == strSinc){
      
      if(bDrop==1){
        sinc.backward(dropout.Get_Out(), _dOut);
      }else{
        sinc.backward(gaffine.Get_Out(), _dOut);
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

  gaffine.gbackward(_X, _dOut);
  dOut = gaffine.Get_dOut();
  dW = gaffine.Get_dW();
  db = gaffine.Get_db();
  
  if(bRand==1){
    dv = gaffine.Get_dv();  
  }

  
  ////////////// Optimization
  
  opt.Set_W(W);
  opt.Set_b(b);
  opt.Set_dW(dW);
  opt.Set_db(db);
  
  if(bRand==1){
    opt.Set_v(v);
    opt.Set_dv(dv);
    
  }
  
  
  opt.Update();
  
  W = opt.Get_W();
  b = opt.Get_b();
  if(bRand==1){
    v = opt.Get_v();
  }
  
  
}





#endif



















