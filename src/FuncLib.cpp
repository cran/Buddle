#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//


//' @keywords internal
// [[Rcpp::export]]
int FindMaxIndex(arma::vec x, double maxVal){
  int r = x.n_elem;
  
  double tmpVal=0;
  int out = 0;
  
  for(int i=1;i<=r;i++){
    tmpVal = x[i-1];
    if(tmpVal == maxVal){
      out = i;
      break;
    }
  }
  return out;
  
}



int GetnEpoch(int y, int x){
  
  int quote = y/x;
  int out = quote;
  if(quote<1){out = 1;}
  return out;
  
}

int GetRemainder(int y, int x){
  int quote = y/x;
  int remainder = y-(quote*x);
  return remainder;
}

arma::vec SMnode(arma::vec y, arma::vec t){
  
  return y-t;
  
}

IntegerVector RandInts(int nMany, int ceiling) {
  
  bool bMode = FALSE;
  if(nMany > ceiling){
    bMode = TRUE;
  }
  IntegerVector results(nMany) ;
  
  IntegerVector frame = seq_len(ceiling) ;
  
  IntegerVector candidate(nMany) ;
  int maxx=ceiling+1;
  
  while (maxx > ceiling) {
    
    candidate = RcppArmadillo::sample(frame, nMany, bMode, NumericVector::create() ) ;
    
    maxx = max(candidate);
    results = candidate;
    
  }
  
  return results;
}




arma::vec BatchNorm(arma::vec x){
  
  int n = x.n_elem;
  
  double muB = sum(x)/n;
  
  arma::vec x_mu = x - muB;
  
  double sigB2 = sum( x_mu % x_mu )/n;
  double eps = 1e-7;
  
  double SigEps = sigB2 + eps;
  
  return x_mu/sqrt(SigEps); 
  
}


arma::mat BatchNorm_Mat(arma::mat X){
  
  int q = X.n_rows;
  int n = X.n_cols;
  
  arma::vec x;
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    x = X.col(i-1);
    out.col(i-1) = BatchNorm(x);
  }
  
  return out;
  
}


arma::vec Original_flVec(arma::vec x){
  
  return exp(-x) / ((1+exp(-x)) % (1+exp(-x)) ) ;
  
}


arma::vec Original_FlVec(arma::vec a){
  arma::vec o1 = a;
  o1.ones();
  
  return (o1 / (1+exp(-a))) ;
}


arma::vec Relu_FlVec(arma::vec a){

  arma::vec ox = a + abs(a);
  
  return (ox / 2) ;
}


arma::vec Relu_flVec(arma::vec x){
  
  int n = x.n_elem;
  arma::vec out(n);
  out.zeros();
  
  for(int i=1;i<=n;i++){
    if(x[i-1]>0){
      out[i-1]=1;
    }
  }
  return out;
  
}



arma::vec FlVec(arma::vec a, int nModel){
  
  int n = a.n_elem;
  arma::vec out(n);
  out.zeros();
  
  if(nModel == 0){
    out = Original_FlVec(a); 
  }else if(nModel == 1){
    out = Relu_FlVec(a); 
  }

  return out;
}


arma::mat FlVec_Mat(arma::mat X, int nModel){
  
  int q = X.n_rows;
  int n = X.n_cols;
  
  arma::vec x;
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    x = X.col(i-1);
    out.col(i-1) = FlVec(x, nModel);
  }
  
  return out;
}




arma::vec flVec(arma::vec x, int nModel){
  
  int n = x.n_elem;
  arma::vec out(n);
  out.zeros();
  
  if(nModel == 0){
    out = Original_flVec(x); 
  }else if(nModel == 1){
    out = Relu_flVec(x); 
  }
  
  return out;
  
}



arma::mat flVec_Mat(arma::mat X, int nModel){
  
  int q = X.n_rows;
  int n = X.n_cols;
  
  arma::vec x;
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    x = X.col(i-1);
    out.col(i-1) = flVec(x, nModel);
  }
  
  return out;
}



double CrossEntropy(arma::vec y, arma::vec t){
  double eps = 1e-7;
  
  return -sum( t %  log(y+eps) ); 
}


arma::vec CrossEntropy_Mat(arma::mat yMat, arma::mat tMat){
  
  //int r = yMat.n_rows;
  int n = yMat.n_cols;
  
  arma::vec y;
  arma::vec t;
  
  arma::vec out(n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    y = yMat.col(i-1);
    t = tMat.col(i-1);
    out[i-1] = CrossEntropy(y, t);
  }
  
  return out; 
}




arma::vec softmax(arma::vec a){
  
  double c = max(a);
  arma::vec exp_a = exp(a-c);
  double sum_exp_a = sum(exp_a);
  
  arma::vec out = exp_a / sum_exp_a;
  
  return out;
}



arma::mat softmax_Mat(arma::mat X){
  
  int q = X.n_rows;
  int n = X.n_cols;
  
  arma::vec x;
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    x = X.col(i-1);
    out.col(i-1) = softmax(x);
  }
  
  return out;
}


arma::vec FlNode(arma::vec SigTilde, arma::vec dOut, int nModel){
  
  int q = SigTilde.n_elem;
  arma::vec tmp = flVec(SigTilde, nModel);
  arma::vec out(q);
  out.zeros();
  
  for(int i=1;i<=q;i++){
    out[i-1] = dOut[i-1] * tmp[i-1];
  }
  
  return out;
}


arma::mat FlNode_Mat(arma::mat SigTilde, arma::mat dOut, int nModel){
  
  int q = SigTilde.n_rows;
  int n = SigTilde.n_cols;
  
  arma::vec si;
  arma::vec di;
  
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    si = SigTilde.col(i-1);
    di = dOut.col(i-1);
    out.col(i-1) = FlNode(si, di, nModel);
  }
  
  return out;
  
}



arma::vec BackwardBatchNorm(arma::vec x, arma::vec dOut){
  
  int n = x.n_elem;
  
  arma::mat GradMat(n,n);
  
  double muB = sum(x)/n;
  
  arma::vec x_mu = x - muB;
  
  double sigB2 = sum( x_mu % x_mu )/n;
  double eps = 1e-7;
  
  double SigEps = sigB2 + eps;
  
  GradMat = ( x_mu*x_mu.t() +  SigEps );
  for(int i=1;i<=n;i++){
    GradMat(i-1,i-1) -= n*SigEps; 
  }
  
  return -1/(n*sqrt(SigEps*SigEps*SigEps)) * GradMat * dOut; 
  
  
}



arma::mat BackwardBatchNorm_Mat(arma::mat X, arma::mat dOut){
  
  int q = X.n_rows;
  int n = X.n_cols;
  
  arma::vec xi;
  arma::vec di;
  
  arma::mat out(q,n);
  out.zeros();
  
  for(int i = 1;i<=n; i++){
    xi = X.col(i-1);
    di = dOut.col(i-1);
    out.col(i-1) = BackwardBatchNorm(xi, di);
  }
  
  return out;
  
}

arma::mat CheckBackwardBatchNorm(arma::vec x){
  
  int n = x.n_elem;
  
  arma::mat GradMat(n,n);
  
  double muB = sum(x)/n;
  
  arma::vec x_mu = x - muB;
  
  double sigB2 = sum( x_mu % x_mu )/n;
  double eps = 1e-7;
  
  double SigEps = sigB2 + eps;
  
  GradMat = ( x_mu*x_mu.t() +  SigEps );
  for(int i=1;i<=n;i++){
    GradMat(i-1,i-1) -= n*SigEps; 
  }
  
  return -1/(n*sqrt(SigEps*SigEps*SigEps)) * GradMat;
  
}





arma::mat InitiateMat(int n, int p){
  
  arma::mat out(n,p);
  double sigInit = 0.01;
  
  for(int i=1;i<=n;i++){
    for(int j=1;j<=p;j++){
      out(i-1,j-1) = R::rnorm(0, sigInit);
    }
  }
  return out;
  
}


arma::vec InitiateVec(int p){
  
  arma::vec out(p);
  double sigInit = 0.01;
  
  for(int j=1;j<=p;j++){
    out[j-1] = R::rnorm(0, sigInit);
  }
  
  return out;
  
}







/////////////////// Moment
arma::mat Momentumv_Mat(arma::mat GradW, arma::mat vMat, double lr){
  double alpha = 0.9;
  
  vMat = alpha*vMat - lr*GradW;
  return vMat;
  
}

/////////////////// Moment
arma::vec Momentumv_Vec(arma::vec GradW, arma::vec vVec, double lr){
  double alpha = 0.9;
  
  vVec = alpha*vVec - lr*GradW;
  return vVec;
  
}


arma::mat MomentumW_Mat(arma::mat W, arma::mat GradW, arma::mat vMat, double lr){
  
  W += vMat;
  return W;
  
}


arma::vec MomentumW_Vec(arma::vec W, arma::vec GradW, arma::vec vVec, double lr){
  W += vVec;
  return W;
  
}


arma::mat AdaGradh_Mat(arma::mat GradW, arma::mat hMat, double lr){
  double delta = 1e-4;
  hMat += GradW % GradW + delta;
  return hMat;
  
}

arma::mat AdaGradW_Mat(arma::mat W, arma::mat GradW, arma::mat hMat, double lr){
  
  W -= lr * (GradW / sqrt(hMat));
  return W;
  
}




arma::vec AdaGradh_Vec(arma::vec GradW, arma::vec hVec, double lr){
  double delta = 1e-4;
  hVec += GradW % GradW + delta;
  return hVec;
  
}


arma::vec AdaGradW_Vec(arma::vec W, arma::vec GradW, arma::vec hVec, double lr){
  
  hVec = AdaGradh_Vec(GradW, hVec, lr);
  W -= lr * (GradW / sqrt(hVec));
  return W;
  
}



arma::mat GetCoeffMat(arma::mat GradW, arma::mat hvMat, double lr, int nMethod){
  
  arma::mat out;
  
  if(nMethod == 1){
    out = AdaGradh_Mat(GradW, hvMat, lr);
  }else if(nMethod == 2){
    out = Momentumv_Mat(GradW, hvMat, lr);
  }
  
  return out;
}


arma::vec GetCoeffVec(arma::vec GradW, arma::vec hvVec, double lr, int nMethod){
  
  arma::vec out;
  
  if(nMethod == 1){
    out = AdaGradh_Vec(GradW, hvVec, lr);
  }else if(nMethod == 2){
    out = Momentumv_Vec(GradW, hvVec, lr);
  }
  
  return out;
}



arma::mat UpdateGradient_Mat(arma::mat W, arma::mat GradW, arma::mat hvMat, double lr, int nMethod){
  
  if(nMethod == 0){
    W -= lr*GradW; 
  }else if(nMethod == 1){
    W = AdaGradW_Mat(W, GradW, hvMat, lr);
  }else if(nMethod ==2){
    W = MomentumW_Mat(W, GradW, hvMat, lr);
  }
  
  return W;
}


arma::vec UpdateGradient_Vec(arma::vec W, arma::vec GradW, arma::vec hvVec, double lr, int nMethod){
  
  if(nMethod == 0){
    W -= lr*GradW; 
  }else if(nMethod == 1){
    W = AdaGradW_Mat(W, GradW, hvVec, lr);
  }else if(nMethod == 2){
    W = MomentumW_Vec(W, GradW, hvVec, lr);
  }
  
  return W;
}



//' @keywords internal
// [[Rcpp::export]]
arma::vec yVec(arma::vec xVec, arma::mat ZList, arma::mat cList, arma::mat ZFinal, arma::vec cFinal, arma::mat W, arma::vec b, int nLayer, int q, int nModel){
  
  int nL2 = nLayer-2;
  
  //int q4 = q*(nL2);
  
  int SInd = 0;
  int EInd = 0;
  int nCount = nL2;
  
  arma::vec mu;
  arma::vec muB;
  arma::vec muTilde;
  
  arma::vec Sig;
  arma::vec SigB;
  arma::vec SigTilde;
  
  arma::vec SigFinal;
  
  arma::mat Z;
  arma::vec c;
  
  
  for(int i=nLayer; i>0; i--){
    
    if(i == nLayer){
      mu = W*xVec + b;
      muB = BatchNorm(mu);
      muTilde = FlVec(muB, nModel);
    }else if(i==1){
      
      SigFinal = ZFinal*SigTilde + cFinal;
      
    }else{
      
      ///////////////////// Extract Z from ZList
      SInd = (nCount-1)*q;
      EInd = SInd + q-1;
      Z = ZList.submat(SInd, 0, EInd, q-1);
      c = cList.col(nCount-1);
      
      nCount -= 1;
      
      if(i== (nLayer-1)){
        Sig = Z*muTilde + c;
      }else{
        Sig = Z*SigTilde + c;
      }
      SigB = BatchNorm(Sig);
      SigTilde = FlVec(SigB, nModel);
      
      
      
    }  
    
  }
  
  arma::vec y = softmax(SigFinal);
  return y;
  
}




arma::mat yVec_Mat(arma::mat Xt, arma::mat ZList, arma::mat cList, arma::mat ZFinal, arma::vec cFinal, arma::mat W, arma::vec b, int nLayer, int q, int nModel){
  
  //int p = Xt.n_rows;
  int n = Xt.n_cols;
  int r = ZFinal.n_rows;
  
  
  arma::mat out(r,n);
  out.zeros();
  
  arma::vec x;
  
  for(int i = 1;i<=n; i++){
    x = Xt.col(i-1);
    out.col(i-1) = yVec(x, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
  }
  
  return out;
  
}




double Loss_Func(arma::vec xVec, arma::vec tVec, arma::mat ZList, arma::mat cList, arma::mat ZFinal, arma::vec cFinal, arma::mat W, arma::vec b, int nLayer, int q, int nModel){
  
  arma::vec y = yVec(xVec, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
  return CrossEntropy(y, tVec);
}


double Loss_Func_Mat(arma::mat Xt, arma::mat tMat, arma::mat ZList, arma::mat cList, arma::mat ZFinal, arma::vec cFinal, arma::mat W, arma::vec b, int nLayer, int q, int nModel){
  
  arma::mat yMat = yVec_Mat(Xt, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
  arma::vec Entropy = CrossEntropy_Mat(yMat, tMat);
  
  double out = mean(Entropy);
  
  return out;
}



arma::vec MainFunc(arma::mat X_Train, arma::mat t_Train, int nMethod, double lr, int nStep, int nLayer, int q, int r, int nModel){
  
  int p = X_Train.n_cols;
  arma::mat X = X_Train.t();
  
  arma::vec tVec(r);
  arma::vec xVec(p);
  
  arma::mat W(q, p);
  arma::mat Z(q,q);
  arma::mat ZFinal(r,q);
  
  
  int nL2 = nLayer-2;
  int q4 = q*nL2;
  
  int SInd = 0;
  int EInd = 0;
  int nCount = nL2;
  
  arma::mat ZList(q4,q);
  
  arma::vec b(q);
  arma::vec c(q);
  arma::vec cFinal(r);
  
  
  ///////////////////////////////////////////  Momentum, AdaGrad 
  
  ///////////////////////////// Momentum     AdaGrad
  arma::mat hvMatW(q,p);
  arma::mat hvMatZ(q,q);
  arma::mat hvMatZFinal(r,q);
  
  hvMatW.zeros();
  hvMatZ.zeros();
  hvMatZFinal.zeros();
  
  
  arma::vec hvVecb(q);
  arma::vec hvVecc(q);
  arma::vec hvVeccFinal(r);
  
  hvVecb.zeros();
  hvVecc.zeros();
  hvVeccFinal.zeros();
  
  ///////////////////////////////////////////////
  
  arma::mat cList(q,nL2);
  
  arma::vec mu(q);
  arma::vec muB(q);
  arma::vec muTilde(q);
  
  arma::vec Sig(q);
  arma::vec SigB(q);
  arma::vec SigTilde(q);
  arma::vec SigFinal(r);
  
  arma::mat SigList(q,nL2);
  arma::mat SigBList(q,nL2);
  arma::mat SigTildeList(q,nL2);
  
  
  ///////////////////////////////////
  arma::vec dOutFinal(r);
  
  arma::vec dOut(q);
  arma::vec dOutFl(q);
  arma::vec dOutB(q);
  
  arma::vec dOutFlmu(q);
  arma::vec dOutBmu(q);
  
  arma::vec GradcFinal(r);
  arma::mat GradZFinal(q,r);
  
  arma::vec Gradc(q);
  arma::mat GradZ(q,q);   
  
  arma::vec Gradb(q);
  arma::mat GradW(p,q); 
  
  
  //////////////////////////////////
  
  arma::vec y(r);
  y.zeros();
  
  arma::vec Loss_Val(nStep);
  Loss_Val.zeros();
  
  for(int k=1; k<=nStep; k++){
    xVec = X.col(k-1);
    tVec = t_Train.col(k-1);
    ////////////////////// Gradient Part
    
    for(int i=nLayer; i>0; i--){
      
      if(i == nLayer){
        if(k==1){
          W = InitiateMat(q, p);
          b = InitiateVec(q);
        }
        mu = W*xVec + b;
        muB = BatchNorm(mu);
        muTilde = FlVec(muB, nModel);
      }else if(i==1){
        if(k==1){
          ZFinal = InitiateMat(r,q);
          cFinal = InitiateVec(r);
          
        }
        SigFinal = ZFinal*SigTilde + cFinal;
        
        
      }else{
        if(k==1){
          Z = InitiateMat(q,q);
          c = InitiateVec(q);
          
        }
        if(i== (nLayer-1)){
          Sig = Z*muTilde + c;
        }else{
          Sig = Z*SigTilde + c;
        }
        SigB = BatchNorm(Sig);
        SigTilde = FlVec(SigB, nModel);
        
        ///////////////////// Save Z in ZList
        SInd = (i-2)*q;
        EInd = SInd + q-1;
        ZList.submat(SInd, 0, EInd, q-1) = Z;
        cList.col(i-2) = c;
        SigList.col(i-2) = Sig;
        SigBList.col(i-2) = SigB;
        SigTildeList.col(i-2) = SigTilde;
        
        nCount -= 1;
        
      } /// End of if i loop  
    
    }  /// End of for i loop
    
    
    y = softmax(SigFinal);
    
    
    
    
    
    for(int i=1;i<=nLayer;i++){
      
      if(i==1){
        dOutFinal = y-tVec;
        GradcFinal = dOutFinal;
        
        SigTilde = SigTildeList.col(0);
        GradZFinal = SigTilde * dOutFinal.t();    //// q-by-r 
        
        hvMatZFinal = GetCoeffMat(GradZFinal.t(), hvMatZFinal, lr, nMethod);
        ZFinal = UpdateGradient_Mat(ZFinal, GradZFinal.t(), hvMatZFinal, lr, nMethod);  /// ZFinal: rx q  
        
        hvVeccFinal = GetCoeffVec(GradcFinal, hvVeccFinal, lr, nMethod);
        cFinal = UpdateGradient_Vec(cFinal, GradcFinal, hvVeccFinal, lr, nMethod);
        
        
      }else if(i==nLayer){
        SInd = (i-3)*q;
        EInd = SInd+q-1;
        Z = ZList.submat(SInd,0,EInd,q-1);
        dOut = Z.t()*dOut;
        
        dOutFlmu = FlNode(muB, dOut, nModel);
        dOutBmu = BackwardBatchNorm(mu, dOutFlmu);
        
        Gradb = dOutBmu;
        GradW = xVec * dOutB.t();
        
        hvMatW = GetCoeffMat(GradW.t(), hvMatW, lr, nMethod);
        W = UpdateGradient_Mat(W, GradW.t(), hvMatW, lr, nMethod);
        
        hvVecb = GetCoeffVec(Gradb, hvVecb, lr, nMethod);
        b = UpdateGradient_Vec(b, Gradb, hvVecb, lr, nMethod);
        
        
      }else{
        
        Sig = SigList.col(i-2);
        SigB = SigBList.col(i-2);
        if(i==(nLayer-1)){
          SigTilde = muTilde;
        }else{
          SigTilde = SigTildeList.col(i-1);
        }
        
        if(i==2){
          dOut = ZFinal.t() * dOutFinal;
        }else{
          SInd = (i-3)*q;
          EInd = SInd+q-1;
          Z = ZList.submat(SInd,0,EInd,q-1);
          dOut = Z.t() * dOut;
        }
        
        dOutFl = FlNode(SigB, dOut, nModel);
        dOutB = BackwardBatchNorm(Sig, dOutFl);
        Gradc = dOutB;
        GradZ = SigTilde * dOutB.t();
        
        hvMatZ = GetCoeffMat(GradZ.t(), hvMatZ, lr, nMethod);
        Z = UpdateGradient_Mat(Z, GradZ.t(), hvMatZ, lr, nMethod);
        
        hvVecc = GetCoeffVec(Gradc, hvVecc, lr, nMethod);
        c = UpdateGradient_Vec(c, Gradc, hvVecc, lr, nMethod);
        
        
        
      }
      
      
    }   //// End of for i loop
  
    Loss_Val[k-1] = Loss_Func(xVec, tVec, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
    
    
  }   ////// End of for k loop
  
  return Loss_Val;
  
}




double GetAccuracy(arma::vec YTest, arma::mat XTest, arma::mat ZList, arma::mat cList, arma::mat ZFinal, arma::vec cFinal, arma::mat W, arma::vec b, int nLayer, int q, int r, int nModel){
  
  int n = XTest.n_rows;
  int p = XTest.n_cols;
  
  arma::rowvec xVec(p);
  arma::vec y(r);
  y.zeros();
  
  double max_yVal = 0;
  int nMaxIndex = 0;
  int nIndex = 0;
  
  double nAccuracy = 0;
  
  for(int i=1;i<=n;i++){
    xVec = XTest.row(i-1);
    y = yVec(xVec.t(), ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
    max_yVal = max(y);
    
    nMaxIndex =  FindMaxIndex(y, max_yVal);
    nIndex = YTest[i-1]+1;
    
    if(nMaxIndex == nIndex){
      nAccuracy += 1;
    }
    
    
  }
  double out = nAccuracy*100/n;

  return out;
  
  
}

//' @keywords internal
// [[Rcpp::export]]
List MainFunc_Mat(arma::vec YTest, arma::mat XTest, arma::vec Y_Train, arma::mat X_Train, arma::mat t_Train, int nN, int nBatch, int nMethod, double lr, int nStep, int nLayer, int q, int r, int nModel){
  
  
  int nEpoch = GetnEpoch(nN, nBatch);
  int nEpochIter = nStep/nEpoch;
  
  arma::vec Train_acc(nEpochIter);
  arma::vec Test_acc(nEpochIter);
  Train_acc.zeros();
  Test_acc.zeros();
  
  int nRem = 0;
  int nEpochInc = 0;
  
  
  int p = X_Train.n_cols;
  
  arma::mat X(nBatch, p);
  arma::mat Xt(p, nBatch);
  arma::mat tMat(r,nBatch);
  
  ////////////////////////////////
  
  
  arma::mat W(q,p);
  arma::mat Z(q,q);
  arma::mat tmpZ(q,q);
  
  arma::mat ZFinal(r,q);
  arma::mat tmpZFinal(r,q);
  
  int nL2 = nLayer-2;
  int q4 = q*nL2;
  
  int SInd = 0;
  int EInd = 0;
  //int nCount = nL2;
  
  arma::mat ZList(q4,q);
  arma::mat tmpZList(q4,q);
  
  arma::vec b(q);
  arma::vec c(q);
  arma::vec tmpc(q);
  arma::vec cFinal(r);
  
  
  ///////////////////////////////////////////  Momentum, AdaGrad 
  
  ///////////////////////////// Momentum     AdaGrad
  arma::mat hvMatW(q,p);
  arma::mat hvMatZ(q,q);
  arma::mat hvMatZFinal(r,q);
  
  hvMatW.zeros();
  hvMatZ.zeros();
  hvMatZFinal.zeros();
  
  
  arma::vec hvVecb(q);
  arma::vec hvVecc(q);
  arma::vec hvVeccFinal(r);
  
  hvVecb.zeros();
  hvVecc.zeros();
  hvVeccFinal.zeros();
  
  arma::mat hvMatZList(q4,q);
  arma::mat hvVeccList(q, nL2);
  
  hvMatZList.zeros();
  hvVeccList.zeros();
  ///////////////////////////////////////////////
  
  arma::mat cList(q,nL2);
  arma::mat tmpcList(q,nL2);
  
  arma::mat mu(q, nBatch);
  arma::mat muB(q, nBatch);
  arma::mat muTilde(q, nBatch);
  
  arma::mat Sig(q, nBatch);
  arma::mat SigB(q, nBatch);
  arma::mat SigTilde(q, nBatch);
  arma::mat SigFinal(r, nBatch);
  
  int BxL2 = nBatch*nL2;
  arma::mat SigList(q,BxL2);
  arma::mat SigBList(q,BxL2);
  arma::mat SigTildeList(q,BxL2);
  
  
  int cS = 0;
  int cE = 0;
  
  //////////////////////////////
  arma::mat dOutFinal(r, nBatch);
  
  
  arma::mat dOut(q, nBatch);
  arma::mat dOutFl(q, nBatch);
  arma::mat dOutB(q, nBatch);
  
  arma::mat dOutFlmu(q, nBatch);
  arma::mat dOutBmu(q, nBatch);
  
  arma::vec GradcFinal(r);
  arma::mat GradZFinal(q,r);
  
  arma::vec Gradc(q);
  arma::mat GradZ(q,q);   
  
  arma::vec Gradb(q);
  arma::mat GradW(p,q);   
  
  
  ///////////////////////////////
  
  arma::vec Loss_Val(nStep);
  Loss_Val.zeros();
  
  IntegerVector SelVec= RandInts(nBatch, nN);
  int nInd = 0;
  
  double dVal=0;
  
  for(int k=1;k<=nStep;k++){

    SelVec= RandInts(nBatch, nN);
    
    if(nBatch != nN){

      for(int j=1;j<= nBatch;j++){
        nInd = SelVec[j-1];
        X.row(j-1) = X_Train.row(nInd-1);
        tMat.col(j-1) = t_Train.col(nInd-1);
      }

    }else{
      X = X_Train;
      tMat = t_Train;
    }
    
    
    Xt = X.t();       /// X: nBatch x p     Xt: p x nBatch
    
    for(int i=nLayer; i>0; i--){
      
      if(i == nLayer){
        if(k==1){
          W = InitiateMat(q, p);
          b = InitiateVec(q);
        }
        mu = W*Xt;                       // mu: q x nBatch
        
        for(int l = 1; l<=nBatch; l++){
          mu.col(l-1) += b;
        }
        
        muB = BatchNorm_Mat(mu);           // muB:     q x nBatch
        muTilde = FlVec_Mat(muB, nModel);  // muTilde: q x nBatch
        
      }else if(i==1){
        if(k==1){
          ZFinal = InitiateMat(r,q);
          cFinal = InitiateVec(r);
          
          tmpZFinal = ZFinal;
          
        }
        SigFinal = ZFinal*SigTilde;    // SigFinal: r x nBatch
        for(int l = 1; l<=nBatch; l++){
          SigFinal.col(l-1) += cFinal;
        }
        
        
        
      }else{
        
        SInd = (i-2)*q;
        EInd = SInd+q-1;
        
        if(k == 1){
          Z = InitiateMat(q,q);
          c = InitiateVec(q);
          
          ZList.submat(SInd,0,EInd,q-1) = Z;
          cList.col(i-2) = c;
          
          tmpZList.submat(SInd,0,EInd,q-1) = Z;
          tmpcList.col(i-2) = c;
        }else{
          
          Z = ZList.submat(SInd,0,EInd,q-1);
          c = cList.col(i-2);
          
        }
        if(i== (nLayer-1)){
          Sig = Z*muTilde;     // Sig: q x nBatch    
        }else{
          Sig = Z*SigTilde;
        }
        
        for(int l = 1; l<=nBatch; l++){
          Sig.col(l-1) += c;
        }
        
        SigB = BatchNorm_Mat(Sig);              // SigB: q x nBatch    
        SigTilde = FlVec_Mat(SigB, nModel);     // SigTilde: q x nBatch    
        
        ///////////////////// Save Z in ZList
        // SInd = (i-2)*q;
        // EInd = SInd + q-1;
        // ZList.submat(SInd, 0, EInd, q-1) = Z;
        // cList.col(i-2) = c;
        
        cS = (i-2)*nBatch;
        cE = cS + nBatch-1;
        
        SigList.submat(0, cS, q-1, cE) = Sig;
        SigBList.submat(0, cS, q-1, cE) = SigB;
        SigTildeList.submat(0, cS, q-1, cE) = SigTilde;
        
        
      } /// End of if i loop  
      
    }  /// End of for i loop
    
    
    arma::mat yMat = softmax_Mat(SigFinal);
    
    
    tmpZList = ZList;
    tmpcList = cList;
    
    tmpZFinal = ZFinal;
    
    for(int i=1;i<=nLayer;i++){
      
      if(i==1){
        dOutFinal = yMat-tMat;       //  dOutFinal : r x nBatch
        GradcFinal = sum(dOutFinal, 1) / nBatch ;   //GradcFinal: rx1      
        
        cS = (2-2)*nBatch;
        cE = cS + nBatch-1;
        
        SigTilde = SigTildeList.submat(0, cS, q-1, cE);
        GradZFinal = SigTilde * dOutFinal.t() / nBatch ;    //// GradZFinal: q x r 
        
        //////////////////// Check nMethod = 0
        
        if(nMethod != 0){
          hvMatZFinal = GetCoeffMat(GradZFinal.t(), hvMatZFinal, lr, nMethod);
        }
        
        ZFinal = UpdateGradient_Mat(ZFinal, GradZFinal.t(), hvMatZFinal, lr, nMethod);   /// ZFinal : rxq
        
        //////////////////// Check nMethod = 0
        if(nMethod != 0){
          hvVeccFinal = GetCoeffVec(GradcFinal, hvVeccFinal, lr, nMethod);
        }
    
        cFinal = UpdateGradient_Vec(cFinal, GradcFinal, hvVeccFinal, lr, nMethod);      /// cFinal : rx1    
        
        
      }else if(i==nLayer){
        SInd = (i-3)*q;
        EInd = SInd+q-1;
        
        tmpZ = tmpZList.submat(SInd,0,EInd,q-1);    //// Previous Z
        dOut = tmpZ.t()*dOutB;                   // dOut: q x nBatch
        
        dOutFlmu = FlNode_Mat(muB, dOut, nModel);       // dOutFlmu: q x nBatch
        dOutBmu = BackwardBatchNorm_Mat(mu, dOutFlmu);  // dOutBmu: q x nBatch
        
        Gradb = sum(dOutBmu,1)/nBatch;
        GradW = Xt * dOutBmu.t()/nBatch;                         // GradW: (px nBatch) x (nBatchxq) = pxq
        
        //////////////////// Check nMethod = 0
        if(nMethod != 0){
          hvMatW = GetCoeffMat(GradW.t(), hvMatW, lr, nMethod);
        }
  
        W = UpdateGradient_Mat(W, GradW.t(), hvMatW, lr, nMethod);
        
        //////////////////// Check nMethod = 0
        if(nMethod != 0){
          hvVecb = GetCoeffMat(Gradb, hvVecb, lr, nMethod);
        }
        
        b = UpdateGradient_Vec(b, Gradb, hvVecb, lr, nMethod);
        
        
      }else{
        
        cS = (i-2)*nBatch;
        cE = cS + nBatch-1;
        
        Sig = SigList.submat(0, cS, q-1, cE);
        SigB = SigBList.submat(0, cS, q-1, cE);
        
        //SigTilde = SigTildeList.submat(0, cS, q-1, cE);
        
        if(i==(nLayer-1)){
          SigTilde = muTilde;
        }else{
          
          cS = (i-1)*nBatch;
          cE = cS + nBatch-1;
          
          SigTilde = SigTildeList.submat(0, cS, q-1, cE);
        }
        
        if(i==2){
          dOut = tmpZFinal.t() * dOutFinal;    ///dOut: q x nBatch
        }else{
          SInd = (i-3)*q;
          EInd = SInd+q-1;
          
          tmpZ = tmpZList.submat(SInd,0,EInd,q-1);    /// previous step Z   
          dOut = tmpZ.t()*dOutB;                     ///dOut: q x nBatch
        }
        
        dOutFl = FlNode_Mat(SigB, dOut, nModel);      //dOutFl: q x nBatch
        dOutB = BackwardBatchNorm_Mat(Sig, dOutFl);   //dOutB: q x nBatch
        Gradc = sum(dOutB, 1)/nBatch ;
        GradZ = SigTilde * dOutB.t() / nBatch;        //GradZ: qxq
        
        SInd = (i-2)*q;
        EInd = SInd+q-1;
        Z = ZList.submat(SInd,0,EInd,q-1);        //// Current Z and c
        c = cList.col(i-2);
        
        
        hvMatZ = hvMatZList.submat(SInd,0,EInd,q-1);
        
        if(nMethod != 0){
          hvMatZ = GetCoeffMat(GradZ.t(), hvMatZ, lr, nMethod);
        }

        hvMatZList.submat(SInd,0,EInd,q-1) = hvMatZ;
        
        Z = UpdateGradient_Mat(Z, GradZ.t(), hvMatZ, lr, nMethod);
        
        hvVecc = hvVeccList.col(i-2);
        
        if(nMethod != 0){
          hvVecc = GetCoeffMat(Gradc, hvVecc, lr, nMethod);
        }

        hvVeccList.col(i-2) = hvVecc;
        
        c = UpdateGradient_Vec(c, Gradc, hvVecc, lr, nMethod);
        
        ///////////////////// Save Z in ZList
        SInd = (i-2)*q;
        EInd = SInd + q-1;
        ZList.submat(SInd, 0, EInd, q-1) = Z;
        cList.col(i-2) = c;
        
        
      }
      
      
    }   //// End of for i loop 1 to nLayer
    
    
    Loss_Val[k-1] = Loss_Func_Mat(Xt, tMat, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, nModel);
    
    nRem = GetRemainder(k, nEpoch);
    
    if(nRem == 0){
      dVal = GetAccuracy(Y_Train, X_Train, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, r, nModel);
      Train_acc[nEpochInc] = dVal;
      
      dVal = GetAccuracy(YTest, XTest, ZList, cList, ZFinal, cFinal, W, b, nLayer, q, r, nModel);
      Test_acc[nEpochInc]  = dVal;
      
      nEpochInc += 1; 
      
    }
    
    
  }
  
  List lst(14);
  
  lst[0] = Loss_Val;
  lst[1] = W;
  lst[2] = b;
  lst[3] = ZList;
  lst[4] = cList;
  lst[5] = ZFinal;
  lst[6] = cFinal;
  lst[7] = Train_acc;
  lst[8] = Test_acc;
  lst[9] = nEpochIter;
  lst[10] = r;
  lst[11] = nLayer;
  lst[12] = q;
  lst[13] = nModel;
  
  
  return lst;
  
  
}





