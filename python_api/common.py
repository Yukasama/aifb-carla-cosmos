"""Gemeinsames fuer run_cosmos.py und run_detector.py: Konfiguration, Pruefungen, Prozessstart.

Wird auch in der Python-3.8-Detektorumgebung importiert: nur Standardbibliothek, keine neuere Syntax.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def file(path, label='Datei', allow_empty=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{label} fehlt (auch defekte Symlinks sind ungueltig): {path}')
    if not allow_empty and path.stat().st_size == 0:
        raise ValueError(f'{label} ist leer: {path}')
    return path.resolve()


def directory(path, label='Ordner', nonempty=False):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'{label} fehlt: {path}')
    if nonempty and not any(path.iterdir()):
        raise ValueError(f'{label} ist leer: {path}')
    return path.resolve()


def new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Ausgabe existiert bereits; neuen Namen verwenden: {path}')
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    directory(parent, 'Ausgabe-Elternordner')
    if not os.access(parent, os.W_OK):
        raise PermissionError(f'Ausgabe nicht schreibbar: {parent}')
    return path


class Config:
    def __init__(self, path):
        import tomllib  # erst ab 3.11, nur im Launcher
        self.path = file(Path(path).expanduser(), 'Pipeline-Konfiguration')
        with self.path.open('rb') as handle:
            self.data = tomllib.load(handle)

    def section(self, name):
        value = self.data.get(name)
        if value is None:
            raise ValueError(f'Konfigurationsabschnitt [{name}] fehlt')
        if not isinstance(value, dict):
            raise TypeError(f'Konfigurationsabschnitt [{name}] muss eine Tabelle sein')
        return value

    def path_value(self, section, key):
        raw = self.section(section).get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'Pfad [{section}].{key} fehlt')
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if '$' in expanded or '://' in expanded:
            raise ValueError(f'Nur aufgeloeste lokale Pfade erlaubt: [{section}].{key} = {raw}')
        path = Path(expanded)
        return path if path.is_absolute() else Path(os.path.abspath(self.path.parent / path))

    def existing(self, section, key, folder=False, nonempty=False):
        path = self.path_value(section, key)
        return directory(path, f'[{section}].{key}', nonempty) if folder else file(path, f'[{section}].{key}')


def add_common_arguments(parser):
    parser.add_argument('--config', default=str(ROOT / 'pipeline.toml'))
    parser.add_argument('--check-only', action='store_true', help='alle Eingaben/Abhaengigkeiten pruefen, ohne GPU-Lauf')


def offline_env():
    # Modelle kommen nur aus den konfigurierten lokalen Ordnern; nie aus dem Netz nachladen.
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', TOKENIZERS_PARALLELISM='false')
    # Weder das venv des Launchers noch ein fremdes Repo auf PYTHONPATH in die Modellumgebungen tragen.
    env.pop('VIRTUAL_ENV', None)
    env.pop('PYTHONPATH', None)
    return env


def gpu_ids(value, section):
    if not isinstance(value, list) or not value or len(set(map(str, value))) != len(value):
        raise ValueError(f'[{section}].gpus muss eine nichtleere Liste eindeutiger GPU-IDs sein')
    return value


def uv_python(project, *uv_args):
    """Interpreter of a uv project; `uv run` installs its locked dependencies on first use."""
    if not shutil.which('uv'):
        raise FileNotFoundError('uv fehlt im PATH (https://docs.astral.sh/uv/)')
    return ['uv', 'run', '--project', str(project), *uv_args, 'python']


def probe(python, modules, cwd, env):
    code = '''import importlib, sys
for name in sys.argv[1:]:
    module = importlib.import_module(name)
    if name == 'numpy' and int(module.__version__.split('.')[0]) >= 2:
        raise RuntimeError('Originale DAIR-Detektor-Konverter benoetigen NumPy < 2; gefunden ' + module.__version__)
    if name == 'pytorch_lightning':
        version = tuple(int(value) for value in module.__version__.split('.')[:2])
        if not (1, 5) <= version < (2, 0):
            raise RuntimeError('Detektor braucht pytorch_lightning >=1.5,<2.0; gefunden ' + module.__version__)
'''
    subprocess.run([*python, '-c', code, *modules], cwd=cwd, env=env, check=True)


def execute(command, cwd, env, output, payloads=None):
    output = new_output(output)
    output.mkdir(parents=True)
    for name, payload in (payloads or {}).items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        write_json(output / name, payload)
    command = list(map(str, command))
    write_json(output / 'command.json', {'argv': command, 'cwd': str(cwd), 'offline': True})
    print('Starte:', subprocess.list2cmdline(command), flush=True)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'Prozess fehlgeschlagen (Exitcode {code}); siehe {output / "run.log"}')


def entrypoint(run, args):
    try:
        run(args)
    except (OSError, TypeError, ValueError, KeyError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'FEHLER: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Abgebrochen', file=sys.stderr)
        return 130
    return 0
