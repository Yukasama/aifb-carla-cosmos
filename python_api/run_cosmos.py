"""Alle Clips aus paths.clips mit Cosmos Transfer 2.5 oder Cosmos 3 uebertragen, offline mit lokalen Gewichten.

uv run run_cosmos.py 2.5 [--check-only]
uv run run_cosmos.py 3 [--check-only]
"""
import argparse
import shutil
from pathlib import Path

from common import (
    Config,
    add_common_arguments,
    directory,
    entrypoint,
    execute,
    file,
    gpu_ids,
    new_output,
    offline_env,
    probe,
    read_json,
    uv_python,
)
from simapi.cosmos import SIZE


def collect_specs(clips_root, version):
    root = directory(clips_root, 'CARLA-Clips', nonempty=True)
    clips = [root] if (root / 'clip.json').is_file() else sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.endswith('.partial') and (p / 'clip.json').is_file())
    if not clips:
        raise ValueError(f'Keine fertigen CARLA-Clips: {root}')
    specs = []
    names = set()
    for clip in clips:
        meta = read_json(file(clip / 'clip.json'))
        if meta.get('holdout'):
            raise ValueError(f'{clip}: Holdout-Clip (Testdaten aus record_scenes.py), laeuft nicht durch Cosmos')
        path = file(clip / 'cosmos' / ('transfer2.5.json' if version == '2.5' else 'cosmos3.json'), 'Cosmos-Spec')
        spec = read_json(path)
        name = spec.get('name')
        if name != clip.name or name in names:
            raise ValueError(f'{path}: eindeutiger name = Clipname erforderlich')
        names.add(name)
        manifest = read_json(file(path.parent / 'manifest.json', 'Padding-Manifest'))
        if manifest['source_frames'] != meta['frames'] or manifest['fps'] != meta['fps']:
            raise ValueError(f'{path}: Manifest stimmt nicht mit Aufnahme ueberein')
        if manifest.get('size') != list(SIZE):
            raise ValueError(f'{path.parent}: Kontrollvideos sind {manifest.get("size")}, Cosmos bekommt {SIZE[0]}x{SIZE[1]}; '
                             'mit python -m simapi.convert --force neu umwandeln')
        if version == '3' and (spec['num_frames'] != manifest['video_frames'] or spec['fps'] != meta['fps']):
            raise ValueError(f'{path}: Spec stimmt nicht mit Kontrollvideos ueberein')
        # Require all exported modalities, even when a specific spec does not enable all of them.
        for stream in ('rgb', 'depth', 'seg', 'vis', 'edge'):
            file(path.parent / (stream + '.mp4'), 'Kontrollvideo')
        for key, value in spec.items():
            if key.endswith(('_path', '_file')) and isinstance(value, str):
                file(path.parent / value, f'Spec-Datei {key}')
            if isinstance(value, dict) and 'control_path' in value:
                file(path.parent / value['control_path'], f'{key}-Kontrolle')
        specs.append(str(path))
    return specs


SIM_TRANSFER_DEFAULTS = '.agents/skills/cosmos3-nano-sim-transfer/inference/assets/sample_args.json'
CONTROLS = {'edge': {'preset_edge_threshold': 'medium'}, 'blur': {'preset_blur_strength': 'medium'},
            'depth': 'depth.mp4', 'seg': 'seg.mp4'}


def sim_transfer_request(spec_path, defaults, control):
    """Request wie prepare_hf_sample.py des Sim-Transfer-Skills: eine Steuerung, ganzer Clip als ein Fenster."""
    folder = Path(spec_path).parent
    spec, manifest = read_json(spec_path), read_json(folder / 'manifest.json')
    frames = manifest['video_frames']
    request = {**defaults, 'name': spec['name'], 'model_mode': 'video2video',
               'prompt': file(folder / 'prompt.json').read_text(encoding='utf-8'), 'fps': float(manifest['fps']),
               'num_frames': frames, 'num_video_frames_per_chunk': frames, 'max_frames': frames,
               'aspect_ratio': spec['aspect_ratio'], 'seed': spec['seed']}
    if isinstance(CONTROLS[control], dict):
        # edge/blur rechnet Cosmos selbst nach dem Skalieren aus dem RGB-Bild (bessere Steuerung)
        request.update({'vision_path': str(folder / 'rgb.mp4'), control: CONTROLS[control]})
    else:
        request[control] = {'control_path': str(folder / CONTROLS[control])}
    return request


def run(args):
    version = args.version
    config = Config(args.config)
    env = offline_env()
    section = 'cosmos25' if version == '2.5' else 'cosmos3'
    settings = config.section(section)
    repo = config.existing(section, 'repo', folder=True)
    uv_args = settings.get('uv', [])
    if not isinstance(uv_args, list) or not all(isinstance(value, str) for value in uv_args):
        raise ValueError(f'[{section}].uv muss eine Liste von uv-Argumenten sein')
    # --frozen: genau das Lockfile des Cosmos-Repos installieren, ohne es zu aendern
    python = uv_python(repo, '--frozen', *uv_args)
    output = new_output(config.path_value(section, 'output'))
    specs = collect_specs(config.path_value('paths', 'clips'), version)
    gpus = gpu_ids(settings.get('gpus'), section)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    payloads = {}
    if version == '2.5':
        entry = file(repo / 'examples/inference.py', 'Transfer-2.5-Entrypoint')
        models = config.existing(section, 'models', folder=True, nonempty=True)
        hub = models / 'hub' if (models / 'hub').is_dir() else models  # HF_HOME- oder HF_HUB_CACHE-Layout
        # Upstream holt jeden Checkpoint per `uvx hf download`; mit HF_HUB_OFFLINE kommt er nur aus diesem Cache.
        if not shutil.which('uvx'):
            raise FileNotFoundError('uvx fehlt im PATH (gehoert zu uv)')
        env.update(HF_HUB_CACHE=str(hub), HUGGINGFACE_HUB_CACHE=str(hub))
        launcher = ['-u'] if len(gpus) == 1 else ['-m', 'torch.distributed.run', '--standalone',
                                                  f'--nproc_per_node={len(gpus)}']
        command = [*python, *launcher, entry, '-i', *specs, '-o', output / 'results', '--no-keep-going']
        if settings.get('disable_guardrails', False):
            command.append('--disable-guardrails')
        modules = ['torch', 'cosmos_transfer2.config']
    else:
        file(repo / 'cosmos_framework/scripts/inference.py', 'Cosmos-3-Entrypoint')
        defaults_file = file(repo / SIM_TRANSFER_DEFAULTS, 'Sim-Transfer-Defaults')
        if len(gpus) != 1:
            raise ValueError('[cosmos3].gpus: das Sim-Transfer-Rezept laeuft auf genau einer GPU')
        control = settings.get('control', 'edge')
        if control not in CONTROLS:
            raise ValueError(f'[cosmos3].control muss eins von {", ".join(CONTROLS)} sein')
        checkpoint = config.existing(section, 'checkpoint', folder=True, nonempty=True)
        file(checkpoint / 'config.json', 'Cosmos-3-Modellkonfiguration')
        if not any(checkpoint.rglob('*.safetensors')) and not any(checkpoint.rglob('*.pt')):
            raise FileNotFoundError(f'Keine Modellgewichte in {checkpoint}')
        for path in checkpoint.rglob('*'):
            if path.is_symlink() or path.is_file():
                file(path, 'Cosmos-3-Checkpoint-Datei')
            if path.name.endswith('.index.json') and path.is_file():
                for shard in set(read_json(path).get('weight_map', {}).values()):
                    file(path.parent / shard, 'Cosmos-3-Checkpoint-Shard')
        tokenizer = config.existing(section, 'text_tokenizer', folder=True, nonempty=True)
        # hf download nvidia/Cosmos3-Nano --include 'text_tokenizer/*' legt die Dateien in text_tokenizer/ ab
        if (tokenizer / 'text_tokenizer').is_dir():
            tokenizer = tokenizer / 'text_tokenizer'
        file(tokenizer / 'tokenizer_config.json', 'Cosmos-3-Text-Tokenizer')
        vae = config.existing(section, 'vae')
        defaults = read_json(defaults_file)
        for spec_path in specs:
            request = sim_transfer_request(spec_path, defaults, control)
            payloads[f'requests/{request["name"]}.json'] = request
        # Wie SIM_TRANSFER_RESOURCE_OVERRIDES/CHECKPOINT_ARGS und der Aufruf im Sim-Transfer-Cookbook
        overrides = ['model.config.tokenizer.bucket_name=', 'model.config.tokenizer.object_store_credential_path_pretrained=',
                     f'model.config.tokenizer.vae_path={vae}', 'model.config.vlm_config.tokenizer.config_variant=hf',
                     f'model.config.vlm_config.tokenizer.pretrained_model_name={tokenizer}',
                     'model.config.vlm_config.pretrained_weights.enabled=False',
                     'model.config.teacher_forcing_frames_per_chunk=1', 'model.config.kv_cache_dtype=null']
        command = [*python, '-m', 'cosmos_framework.scripts.inference',
                   '-i', *(output / name for name in payloads), '-o', output / 'results',
                   '--checkpoint-path', checkpoint, '--config-file', checkpoint / 'config.json',
                   '--credential-path', '', '--experiment-overrides', *overrides,
                   '--defaults-file', defaults_file, '--autoregressive=True',
                   '--cp-size', '1', '--cfgp-size', '1', '--dp-shard-size', '1', '--dp-replicate-size', '1',
                   '--num-outputs', '1', '--max-num-seqs', '1',
                   '--num-steps', '4', '--guidance', '1', '--control-guidance', '1',
                   '--kv-cache-inference-size', '0', '--attention-sink-size', '0',
                   '--no-use-torch-compile', '--no-use-cuda-graphs', '--no-condition-stream-encode',
                   '--no-diffusion-cache', '--prompt-upsampling=False']
        env.update(COSMOS_INTERNAL='0', COSMOS_TRAINING='1', WANDB_MODE='disabled')
        env.pop('COSMOS_DISTILLATION_IMPL', None)
        modules = ['torch', 'cosmos_framework']
    probe(python, modules, repo, env)
    print(f'Preflight OK: {len(specs)} Clips, Cosmos {version}, Ausgabe {output}')
    if args.check_only:
        return
    execute(command, repo, env, output, payloads)
    # Some upstream batch entrypoints continue after failed samples; do not report partial success.
    from simapi.dataset import frames_of
    for spec_path in specs:
        spec = read_json(spec_path)
        manifest = read_json(Path(spec_path).parent / 'manifest.json')
        results = output / 'results'
        result = results / (spec['name'] + '.mp4') if version == '2.5' else results / spec['name'] / 'vision.mp4'
        frames_of(str(file(result, 'Cosmos-Ergebnis')), {0}, manifest['video_frames'], manifest['fps'])
    print(f'Alle {len(specs)} Ergebnisse vollstaendig geprueft.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('version', choices=('2.5', '3'), help='Cosmos Transfer 2.5 oder Cosmos 3')
    add_common_arguments(parser)
    raise SystemExit(entrypoint(run, parser.parse_args()))
