# RFs_Calculation

+ 关于卷积的实现细节可以参考：
  - [1]. Vincent Dumoulin, Francesco Visin - A guide to convolution arithmetic for deep learning (BibTeX)
  - [2].<a href="https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807" target="_blank">A guide to receptive field arithmetic for Convolutional Neural Networks</a> 
  - [3]. <a href="https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51" target="_blank">computeReceptiveField.py</a>
  
+ 在引用[3]的基础上，做了一些修改，以使得其能适应dilated convolution以及具有residual blocks的backbone。并且以darknet-53为例，计算感受野(Receptive field)。<\br>
 
 - CNNs的感受野大，但是不代表他的有效感受野也很大，甚至可能很小，而这个effective receptive field才是决定CNNs能力的关键。可以参考论文：
  - [4]. <a href="https://arxiv.org/abs/1701.04128" target="_blank">Understanding the Effective Receptive Field in Deep Convolutional Neural Networks
</a>
 

