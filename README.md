## Introductions:

**Establishing an AI-based Forward-Reverse engineering platform for rational designing cRBSs of transcription factor biosensor in _Escherichia coli_**

&ensp;&ensp;We report a novel AI-based forward-reverse engineering framework for *de novo* cRBS design with the selected TFB dynamic range in *E. coli*. The framework had a superior capacity to process the imbalanced minority-class datasets and could capture the correlation between the cRBS and dynamic range to mimic the design of cRBS with the selected TFB dynamic range based on the WGAN-GP and the BAGAN-GP models. From the experimental results of forward and reverse engineering, up to 86% and 97% of the AI-designed cRBSs were experimentally demonstrated to be functional, respectively. Here, we introduced the code used for cRBS sequences generation with desired TFB dynamic range

## Prerequisites:

- Python ==  3.8.3
- TensorFlow == 2.2.0
- CUDA == 11.3
- Anaconda == 4.8.5
- sklearn, scipy, numpy, matplotlib
- A recent NVIDIA GPU

## Features:

- WGAN-GP model to generate cRBS sequence data

- Convolutional neural network (CNN) model to predict the TFB output of the WGAN-GP generated cRBSs under ON and OFF state

- BAGAN_GP model to design novel cRBSs with desired TFB dynamic range

- Filter model to reduce the prediction noise of BAGAN-GP model

## References

[1] Ding, N.N., Yuan, Z.Q., Zhang, X.J., Chen, J., Zhou, S.H. and Deng, Y. (2020) Programmable cross-ribosome-binding sites to fine-tune the dynamic range of transcription factor-based biosensor. *Nucleic Acids Res*, **48**, 10602-10613.

[2] Wang, Y., Wang, H., Wei, L., Li, S., Liu, L. and Wang, X. (2020) Synthetic promoter design in Escherichia coli based on a deep generative network. *Nucleic Acids Res*, **48**, 6403-6412.

[3] Huang, G. and Jafari, A.H. (2021) Enhanced balancing GAN: minority-class image generation. *Neural Computing and Applications*, 1-10.

[4] Sanchez-Lengeling, B. and Aspuru-Guzik, A. (2018) Inverse molecular design using machine learning: Generative models for matter engineering. *Science*, **361**, 360-365.
