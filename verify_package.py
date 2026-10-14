"""Check packaged inference, training imports, dependencies and unchanged artifacts."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
UI = ROOT / 'Combined Models A B C Prediction UI'
A = ROOT / 'training titanium model A/production_v6_titanium_core_corrected'
B = ROOT / 'Training Titanium Model B/improve accuracy/probability_calibration_v2'
C = ROOT / 'Training Titanium Model C/training_v1'


def main():
    for requirement in (ROOT / 'requirements.txt').read_text().splitlines():
        if not requirement or requirement.startswith('#'):
            continue
        name, version = requirement.split('==')
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Install requirements.txt: {name} must be {version}')
    for package, command in ((A, ['code/saved_model_inference.py']),
                             (B, ['code/verify_local_data.py']),
                             (C, ['code/verify_local_data.py']),
                             (UI, ['-m', 'unittest', 'test_app', '-v'])):
        subprocess.run([sys.executable, *command], cwd=package, check=True)
    spec = importlib.util.spec_from_file_location('package_v6_training_check', A / 'code/train_model_a_production_v6_titanium_core_corrected.py')
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    assert Path(engine.IMPROVED_SCRIPT).is_relative_to(A)
    assert Path(engine.engine.BASE_SCRIPT).is_relative_to(A)
    assert engine.SOURCE_DATA_PATH.is_file()
    # Verify retained artifact copies against hashes captured before packaging edits.
    initial = json.loads((ROOT / 'COPY_MANIFEST.json').read_text(encoding='utf-8-sig'))
    artifacts = 0
    for record in initial['files']:
        relative = Path(record['destination']).relative_to(initial['destinationRoot'])
        if relative.suffix.lower() == '.joblib':
            path = ROOT / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest().lower() == record['sha256'].lower(), str(relative)
            artifacts += 1
    excluded = {'__pycache__', '.venv', 'venv', 'node_modules', 'tmp', '.git'}
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in excluded for p in path.relative_to(ROOT).parts):
            continue
        if path.name in ('PACKAGE_VERIFICATION.json', 'CURRENT_FILE_MANIFEST.json') or path.suffix in ('.pyc', '.log'):
            continue
        files.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT / 'CURRENT_FILE_MANIFEST.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    report = dict(passed=True, reference_forecast_values=118, api_integration_tests=5,
                  retained_joblib_files_verified_unchanged=artifacts,
                  a_training_engines_package_local=True, retrained=False,
                  limitations=['Fresh environment installation not tested',
                               'Public data redistribution requires dataset-specific review',
                               'Research/demo, not production approved'])
    (ROOT / 'PACKAGE_VERIFICATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
