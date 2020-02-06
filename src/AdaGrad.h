
#ifndef __SGD_H
#define __SGD_H


class SGD{
  
private:
  
  int nHiddenLayer;     
  double d_learning_rate;
  
  
public:
  
  SGD(){
    nHiddenLayer=1;     
    d_learning_rate=0;
  }
  
  SGD(int _nHiddenLayer, double _d_learning_rate) { // Default matrix member variable initialization
    
    nHiddenLayer = _nHiddenLayer;     
    d_learning_rate = _d_learning_rate;
    
  }

  void Update(Wb* Arr_Wb, Wb* Arr_dWb);
  
};


void SGD::Update(Wb* Arr_Wb, Wb* Arr_dWb){
  
  for(int i=1;i<=(nHiddenLayer+1);i++){

    Arr_Wb[i-1].Set_W( Arr_Wb[i-1].Get_W() - d_learning_rate* Arr_dWb[i-1].Get_W());
    Arr_Wb[i-1].Set_b( Arr_Wb[i-1].Get_b() - d_learning_rate* Arr_dWb[i-1].Get_b());

  }


  
}



#endif



















