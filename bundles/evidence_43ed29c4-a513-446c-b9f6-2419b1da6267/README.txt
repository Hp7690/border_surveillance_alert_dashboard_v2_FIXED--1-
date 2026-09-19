Offline verification:
  pip install cryptography
  python verify_bundle.py <this-unzipped-folder>
The script recomputes every block hash + Ed25519 signature and the SHA-256 of each evidence file, without contacting the IBVAP server.
