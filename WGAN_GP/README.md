## Using WGAN-GP model to generate cRBS sequence data

- Procdure:

  1. Store the cRBS_long sequence data in “.\seq\cRBS_long.txt”
  2. Change the setting in WGAN-GP.py reported by Wang *et al*., including ITERS, SEQ_LEN, CRITIC_ITERS, and MAX_N_EXAMPLES
  3. Train the model by python WGAN-GP.py
- The generated cRBS sequences will be saved in the current folder