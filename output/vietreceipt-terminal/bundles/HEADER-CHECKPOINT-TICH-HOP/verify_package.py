"""Verify bundled source, weights and samples without importing OCR models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'PACKAGE-MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['files_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(name + ': missing or outside bundle')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name + ': hash mismatch')
    print(json.dumps({'status':'FAIL' if errors else 'PASS', 'verified_files':len(manifest['files_sha256']), 'errors':errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
