## Training Convolutional neural network (CNN) model to predict the TFB output of the WGAN-GP generated cRBSs under ON and OFF state

The CNN model was trained by the dataset from the Ding et al which contains 7053 cRBSs with corresponding TFB output constructed by FACS based on random and normal distribution under ON (20 g/l glucarate) and OFF (0 g/l glucarate) state.

- Procdure:

  1. Store the predicted cRBSs sequence in “.\seq\WGAN-GP_generated_cRBSs.fas”
  2. Use the file convertor by python csv_format_convert_to_npy_format.ipynb to convert the csv file to npy file for obtaining the training datasets of four predictive models
  3. Use the predictor by python Training_four_predictive_models.ipynb and Predictor_models.py to train the CNN model and predict the TFB output of a given cRBS under ON and OFF state
- The predicted results will be saved in Accuracy_seq_Flu_MSE.txt