   rm -rf espnet-Master  # Remove conflicting directory (if necessary)
   tar -xzvf espnet-master_copy.tar.gz  # Extract pre-downloaded archive (replace with your archive name)
   pip3 install espnet espnet2  # Install required libraries (adjust based on your needs)
   pip3 uninstall espnet espnet2 # Remove repo names so the conflicting version from /envs folder is not used but the modified.
   cd espnet-master_copy/egs2/librispeech_100/asr1/  # Navigate to the recipe directory
   # Manually edit db.sh to set the LIBRISPEECH download directory path
   # Open db.sh, locate the download directory line, edit the path, and save the changes.
   ./run.sh  # Run the recipe (assuming run.sh has execute permissions)

   # There will be some libraries that will need to be installed manually, such as typecheck and sclite but just proceed with installing and re-running script until all works.
