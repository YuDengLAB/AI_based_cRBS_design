## Training filter model to reduce the prediction noise of BAGAN-GP model

The filter model was trained by the dataset from the Ding et al which contains 7053 cRBSs with five TFB dynamic range measured by FACS.

- Procdure:

  1. Store the trained cRBSs sequence in 
     > ./seq_1.csv # sublibrary-Ⅰ
     > ./seq_2.csv # sublibrary-Ⅱ 
  2. Use the predictor by python predict_test_short_sublibrary.ipynb to train the CNN model and predict the TFB dynamic range of a given cRBS sequence
- The predicted results will be saved in the current folder and used to reduce the prediction noise of the BAGAN-GP model.