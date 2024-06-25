## Using a Modified espnet Implementation for ASR

**Requirements:**

* Git (https://www.git-scm.com/downloads)
* GNU tar (https://man7.org/linux/man-pages/man1/tar.1.html)
* Python 3 (https://www.python.org/downloads/)

**Hardware:**

A computer with sufficient processing power (CPU and potentially GPU) is recommended for audio processing and deep learning (depending on the model and dataset).

**Steps:**

1. **Clone, Prepare, and Run the Recipe:**
   
   ```bash
   git clone  https://github.com/Jezidmar/Modified_implementation.git  # Clone the repository
   pip install espnet  # Install required libraries (adjust based on your needs)
   pip install torchaudio nnAudio spafe librosa tensorboard wandb
   cd espnet/egs2/librispeech_100/asr1/  # Navigate to the recipe directory
   # To install scoring library go to ../espnet/tools/installers and run install_sctk.sh
   # Manually edit db.sh to set the LIBRISPEECH download directory path 
   # Edit path.sh file and set PYTHONPATH=$MAIN_ROOT
   # Inside asr.sh file configure nj option to the one that suits your config.
   # inside run.sh file, set stage to 1 and configure number of gpus to use.
   ./run.sh  # Run the recipe (assuming run.sh has execute permissions)
   # After running recipe and reaching STAGE 10, stop the run and run following:  pip uninstall espnet 
   # Re-run the run.sh script and proceed with steps.
   # You may need to login to wandb to run recipes. Otherwise, go to /conf/{trial.yaml} and remove lines related to wandb


///
   **Modify features used for extraction:**
   ```bash
   /espnet2/layers/log_mel.py  # Here are the feature classes
           /asr/frontend/default.py # Here one can change the features to extract by modifying DefaultFrontend class 
