# Synthetic_cRBS_design
Synthetic cRBS design in Escherichia coli based on deep learning

## Prerequisites

- Python ==  3.8.8
- TensorFlow == 2.6.0
- CUDA == 11.5
- Anaconda == 4.10.1
- A recent NVIDIA GPU

## Using BAGAN_GP model to design novel cRBSs with desired TFB dynamic range

- 1. Store the train sequence data in ".\data\raw_data\" file. The structure of training set as:

     > cRBS_short.csv   # Include all sequence data
     >
     > 1.txt                      # Class one sequence
     >
     > 2.txt
     >
     > 3.txt
     >
     > ....

- 2. Change the setting in prepare_data.py and bagan_gp_model.py, which may include BATCH_SIZE, n_classes(based on you data structure to change), etc

- 3. Data preprocessing and train the model  by

     ``` shell
     python prepare_data.py
     python Encodermodel.py
     python bagan_gp_model.py
     ```

- 4. Model weights and output are stored in the model folder

- 5. The APIs used to analyze the model are encapsulated in base_bagan_create_seq_and_analyse.ipynb, used by

     ``` shell
     jupyter notebook base_bagan_create_seq_and_analyse.ipynb
     ```

     
