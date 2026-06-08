#!/usr/bin/env python
# coding: utf-8

# In[4]:


import subprocess
import sys

def run_script(command):
    """Utility function to run a command and handle errors."""
    print(f"Executing: {' '.join(command)}")
    try:
        # check=True raises an error if the script fails (non-zero exit code)
        subprocess.run(command, check=True)
        print("Success!\n" + "-" * 40)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {' '.join(command)}")
        sys.exit(e.returncode)

def main():
    print("Starting pipeline...\n" + "=" * 40)

    # --- Early Setup / Initialization Steps ---

    # 1. Compile Ab Initio Configs
    run_script(["python", "scripts_abinitio/compile_abinitio_configs.py"])

    # 2. True Interface Restraints (Oracle)
    run_script(["python", "scripts_oracle/true_interface_restraints.py"])

    # 3. Compile Oracle Configs
    run_script(["python", "scripts_oracle/compile_oracle_configs.py"])

    # --- Main Inference & Evaluation Pipeline ---

    # 4. Inference with ESM3-PPISites large model
    # Input: esm3_reps/db5_embeddings.pt -> Output: inference_results/db5_results.csv
    run_script(["python", "scripts_eval/inferenceESM3.py"])

    # 5. Patches creation
    # Input: inference_results/db5_results.csv -> Output: inference_results/db5_for_pairing.csv
    run_script(["python", "scripts_eval/PATCHES.py"])

    # 6. Pairing patches and patches config file creation
    # Input: inference_results/db5_for_pairing.csv
    # Output: haddock-units/*/tbls and haddock-units/*/configs
    run_script(["python", "scripts_eval/PAIRING.py"])

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




