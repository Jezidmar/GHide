# Espnet paper AAAI
For paper implementations

## Using a Modified espnet Implementation for ASR



**Requirements:**

* Git (https://www.git-scm.com/downloads)
* GNU tar (https://man7.org/linux/man-pages/man1/tar.1.html)
* Python 3 (https://www.python.org/downloads/)
* espnet toolkit (https://github.com/espnet)

**Hardware:**

A computer with sufficient processing power (CPU and potentially GPU) is recommended for audio processing and deep learning (depending on the model and dataset).

**Dataset:**

This implementation likely uses the LIBRISPEECH dataset, a large collection of audiobooks for speech recognition training (https://openslr.org/resources.php). You'll need to download this dataset separately.

**Steps:**

1. **Clone, Prepare, and Run the Recipe:**
   ```bash
   git clone -b master https://github.com/Jezidmar/Modified_implementation.git  # Clone the repository
   rm -rf espnet-Master  # Remove conflicting directory (if necessary)
   tar -xzvf espnet-master_copy.tar.gz  # Extract pre-downloaded archive (replace with your archive name)
   pip3 install espnet espnet2  # Install required libraries (adjust based on your needs)
   pip3 uninstall espnet espnet2 # Remove repo names so the conflicting version from /envs folder is not used but the modified.
   cd espnet-master_copy/egs2/librispeech_100/asr1/  # Navigate to the recipe directory
   # Manually edit db.sh to set the LIBRISPEECH download directory path
   # Open db.sh, locate the download directory line, edit the path, and save the changes.
   ./run.sh  # Run the recipe (assuming run.sh has execute permissions)
   # There will be some libraries that will need to be installed manually, such as typecheck and sclite but just proceed with installing and re-running script until all works. 

