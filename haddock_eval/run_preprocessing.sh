set -euo pipefail

python src/structure_io.py
python src/mappings.py
python src/oracle_setup_flex.py
python src/inference_call.py
python src/parse_call.py
python src/make_patches.py
python src/do_pairing_noflex.py

