# RFs_Calculation

+ 关于卷积的实现细节可以参考：
  + [1]. Vincent Dumoulin, Francesco Visin - A guide to convolution arithmetic for deep learning (BibTeX)
  + [2]. A guide to receptive field arithmetic for Convolutional Neural Networks
  + [3].(computeReceptiveField.py)( https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51)
  
+ 在引用[3]的基础上，做了一些修改，以使得其能适应dilated convolution以及具有residual blocks的backbone。并且以darknet-53为例，计算感受野(Receptive field)的。
 
 

