set -euo pipefail

python3 src/structure_io.py
python3 src/mappings.py
python3 src/oracle_setup.py
python3 src/inference_call.py
python3 src/parse_call.py
python3 src/make_patches.py
python3 src/do_pairing_noflex.py

