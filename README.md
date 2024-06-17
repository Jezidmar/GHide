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

(1) git clone -b master https://github.com/Jezidmar/Modified_implementation.git //
(2) rm -rf espnet-Master \\
(3) tar -xzvf espnet-master_copy.tar.gz
(4) cd espnet-master_copy/egs2/librispeech_100/asr1/
(5) pip3 install espnet & pip3 install espnet2
(6) pip3 uninstall espnet & pip3 uninstall espnet2
(7) Modify db.sh s.t. you set the download directory for LIBRISPEECH dataset
(8) Run recipe via run.sh. 
    - asr_config.yaml is in /conf folder
    - decoder_asr.yaml is in /conf folder
    - modify number of gpus with --ngpu argument


