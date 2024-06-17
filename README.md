# GHide
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

1. **Clone the Modified Repository:**

```bash
git clone -b master [https://github.com/Jezidmar/Modified_implementation.git](https://github.com/Jezidmar/Modified_implementation.git)



(1) git clone -b master https://github.com/Jezidmar/Modified_implementation.git
(2) rm -rf espnet-Master
(3) tar -xzvf espnet-master_copy.tar.gz
(4) cd espnet-master_copy/egs2/librispeech_100/asr1/
(5) pip3 install espnet & pip3 install espnet2
(6) pip3 uninstall espnet & pip3 uninstall espnet2
(7) Modify db.sh s.t. you set the download directory for LIBRISPEECH dataset
(8) Run recipe via run.sh. 
    - asr_config.yaml is in /conf folder
    - decoder_asr.yaml is in /conf folder
    - modify number of gpus with --ngpu argument




# Steps 2-4: Download and Extract (assuming pre-downloaded archive)
rm -rf espnet-Master  # Remove conflicting directory (if necessary)
tar -xzvf espnet-master_copy.tar.gz  # Extract pre-downloaded archive

# Step 5: Install/Uninstall Espnet (Optional)
# Choose the lines you need based on your setup:
pip3 install espnet  # Install espnet
# pip3 uninstall espnet  # Uninstall espnet (if needed)
pip3 install espnet2  # Install espnet2 (if needed)
# pip3 uninstall espnet2  # Uninstall espnet2 (if needed)

# Step 6: Modify `db.sh` Script (Manual Configuration)
# - Open `db.sh` in a text editor.
# - Locate the LIBRISPEECH download directory line.
# - Edit the path to your desired download location.
# - Save the changes.

# Step 7: Run the Recipe
./run.sh  # Assuming run.sh has execute permissions

# Step 8: Using Multiple GPUs (Optional)
# ./run.sh --ngpu <number_of_gpus>  # Specify the number of GPUs

