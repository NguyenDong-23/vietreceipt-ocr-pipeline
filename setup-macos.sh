#!/usr/bin/env bash
set -euo pipefail
python3.12 -m venv output/pp-doclayout-v3-1511/.venv
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install --upgrade pip
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install -r output/pp-doclayout-v3-1511/requirements-layout.txt
output/pp-doclayout-v3-1511/.venv/bin/python -m pip install -r requirements-document-reader.txt
python3.12 -m venv output/vietreceipt-terminal/.venv-footer
output/vietreceipt-terminal/.venv-footer/bin/python -m pip install --upgrade pip
output/vietreceipt-terminal/.venv-footer/bin/python -m pip install -r output/vietreceipt-terminal/bundles/footer/requirements-footer.txt
swiftc -O output/vietreceipt-terminal/header_vision.swift -o output/vietreceipt-terminal/header_vision
