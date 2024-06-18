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
   git clone  [https://github.com/Jezidmar/Modified_implementation.git](https://github.com/Jezidmar/Modified_implementation.git)  # Clone the repository
   pip3 install espnet espnet2  # Install required libraries (adjust based on your needs)
   cd espnet/egs2/librispeech_100/asr1/  # Navigate to the recipe directory

   # Manually edit db.sh to set the LIBRISPEECH download directory path
   # Edit path.sh file to set your /path/to/espnet
   # inside run.sh file, set stage to 1 and configure number of gpus to use.
   ./run.sh  # Run the recipe (assuming run.sh has execute permissions)
