"""Roadside-Detektor (BEVHeight++, CoBEV, HeightFormer) trainieren oder auf echtem DAIR-V2X-I evaluieren.

uv run run_detector.py train --data cosmos --cosmos 2.5 --name bh-c25 [--model cobev] [--fullhd] [--resume last.ckpt]
uv run run_detector.py evaluate --checkpoint .../last.ckpt --name bh-c25-on-dair [--model cobev]

Trainingsbilder sind 1280 x 720, mit --fullhd 1920 x 1080; echtes DAIR-V2X-I ist 1920 x 1080.
Fehlt der Datensatz (Variante unter paths.datasets bzw. [dair].prepared), wird er vorher gebaut. Die
Unterbefehle _prepare und _detector startet das Skript selbst in der Detektor-Umgebung (Python 3.8):
deshalb oben nur Standardbibliothek und keine Syntax neuer als 3.8.
"""
import argparse
import ast
import copy
import importlib.util
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from common import (
    ROOT,
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
    write_json,
)

SELF = Path(__file__).resolve()
DETECTOR_ENV = ROOT / 'envs' / 'detector'
MODELS = ('bevheight++', 'cobev', 'heightformer')
HD, FULLHD = (1280, 720), (1920, 1080)  # FULLHD = Bildgroesse von DAIR-V2X-I
CONVERTERS = ('scripts/gen_info_dair.py', 'scripts/data_converter/gen_kitti/label_lidarcoord_to_cameracoord.py',
              'scripts/data_converter/gen_kitti/label_json2kitti.py')


def safe_name(value):
    if not value or not all(c.isalnum() or c in '-_.' for c in value) or value in ('.', '..'):
        raise ValueError('Name darf nur Buchstaben, Ziffern, Punkt, Unterstrich und Bindestrich enthalten')
    return value


def dataset_key(data, cosmos, fullhd):
    return ('carla' if data == 'carla' else 'cosmos' + cosmos.replace('.', '')) + ('-fullhd' if fullhd else '-hd')


def split_ids(path):
    split = read_json(file(path, 'DAIR-Split'))
    if 'infrastructure_split' in split:
        split = split['infrastructure_split']
    for key in ('train', 'val'):
        ids = split.get(key)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f'{path}: Split {key} fehlt oder ist leer')
        if any(not isinstance(sid, str) or len(sid) != 6 or not sid.isascii() or not sid.isdigit() for sid in ids):
            raise ValueError(f'{path}: {key} muss sechsstellige DAIR-IDs als Strings enthalten')
        if len(set(ids)) != len(ids):
            raise ValueError(f'{path}: doppelte IDs in {key}')
    if set(split['train']) & set(split['val']):
        raise ValueError(f'{path}: Training und Validierung ueberschneiden sich')
    return split


def validate_raw(root, split_path):
    root = directory(root, 'DAIR-Rohdaten', nonempty=True)
    split = split_ids(split_path)
    entries = read_json(file(root / 'data_info.json', 'DAIR data_info.json'))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{root}: leere/ungueltige data_info.json')
    by_id = {}
    fields = ('image_path', 'pointcloud_path', 'calib_camera_intrinsic_path',
              'calib_virtuallidar_to_camera_path', 'label_camera_std_path')
    for item in entries:
        sid = Path(item['image_path']).stem
        if sid in by_id:
            raise ValueError(f'{root}: doppelte Bild-ID {sid}')
        by_id[sid] = item
        for key in fields:
            value = item.get(key)
            if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
                raise ValueError(f'{root}: ungueltiger relativer Pfad in {key}')
            file(root / value, f'DAIR {sid}: {key}')
        # Official loaders construct these canonical paths instead of using data_info.
        for relative in (f'image/{sid}.jpg', f'velodyne/{sid}.pcd', f'label/camera/{sid}.json',
                         f'calib/camera_intrinsic/{sid}.json', f'calib/virtuallidar_to_camera/{sid}.json'):
            file(root / relative, 'DAIR-Loader-Eingabe')
    missing = (set(split['train']) | set(split['val'])) - set(by_id)
    if missing:
        raise FileNotFoundError(f'{root}: Split-IDs fehlen in data_info.json: {sorted(missing)[:10]}')
    return split


def prepared_manifest(path):
    manifest = read_json(file(Path(path) / 'prepared.json', 'Vorbereiteter Datensatz'))
    root = directory(manifest['raw_root'], 'Vorbereitete DAIR-Rohdaten')
    split = validate_raw(root, manifest['split'])
    for name in ('train_info', 'val_info'):
        file(manifest[name], f'Vorbereitete {name}')
    label_dir = directory(manifest['gt_labels'], 'KITTI-Ground-Truth')
    for sid in split['train'] + split['val']:
        file(label_dir / (sid + '.txt'), 'KITTI-Label', allow_empty=True)
    return manifest


def existing_dataset(path):
    """Manifest eines fertigen Datensatzes, oder None, wenn er erst gebaut werden muss."""
    if (path / 'prepared.json').is_file():
        return prepared_manifest(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Datensatz ohne prepared.json (abgebrochener Aufbau?); pruefen und loeschen: {path}')
    new_output(path)
    return None


def synthetic_sources(config, data, cosmos, size):
    """Prueft alle Quellen einer Trainingsvariante, bevor ein womoeglich grosser Datensatz entsteht."""
    clips_root = config.existing('paths', 'clips', folder=True, nonempty=True)
    from simapi import dataset
    from simapi.convert import find_clips
    clips = list(find_clips([str(clips_root)]))
    if not clips:
        raise ValueError('Keine fertigen Clips fuer den Trainingsdatensatz')
    cosmos_dirs = []
    if data == 'cosmos':
        section = 'cosmos25' if cosmos == '2.5' else 'cosmos3'
        results = directory(config.path_value(section, 'output') / 'results', 'Cosmos-Ergebnisse', nonempty=True)
        cosmos_dirs = [str(results)]
    for clip in clips:
        if read_json(file(Path(clip) / 'clip.json')).get('holdout'):
            raise ValueError(f'{clip}: Holdout-Clip (Testdaten aus record_scenes.py), gehoert nicht ins Training')
        meta = read_json(file(Path(clip) / 'dair/meta.json', 'DAIR-Cliplabels'))
        file(Path(clip) / 'dair/calib/camera_intrinsic.json')
        file(Path(clip) / 'dair/calib/virtuallidar_to_camera.json')
        for i in meta['frames']:
            file(Path(clip) / f'dair/label/{i:04d}.json')
            file(Path(clip) / f'dair/velodyne/{i:04d}.pcd')
        if data == 'carla':
            file(Path(clip) / 'cosmos/rgb.mp4', 'CARLA-RGB-Video')
            exported = read_json(file(Path(clip) / 'cosmos/manifest.json', 'Padding-Manifest')).get('size', [0, 0])
            if exported[0] < size[0]:
                raise ValueError(f'{clip}: cosmos/rgb.mp4 ist {exported[0]}x{exported[1]}, kleiner als die Trainingsgroesse '
                                 f'{size[0]}x{size[1]}; mit python -m simapi.convert --force neu umwandeln')
        for result_root in cosmos_dirs:
            if dataset.cosmos_result(result_root, Path(clip).name) is None:
                raise FileNotFoundError(f'Cosmos-Ergebnis fehlt: {result_root}/{Path(clip).name}')
    settings = config.section('dataset')
    val, seed = settings.get('val_fraction', 0.2), settings.get('seed', 0)
    if not 0 < val < 1:
        raise ValueError('[dataset].val_fraction muss zwischen 0 und 1 liegen')
    return clips, cosmos_dirs, val, seed


def prepare(python, repo, env, raw_root, split_path, output, synthetic):
    """Offizielle Info-Pickles und KITTI-Labels, erzeugt in der Detektor-Umgebung (_prepare)."""
    request = {'repo': str(repo), 'raw_root': str(raw_root), 'split': str(split_path), 'output': str(output),
               'synthetic': synthetic}
    execute([*python, '-u', SELF, '_prepare', output / 'request.json'], repo, env, output, {'request.json': request})


def build_synthetic(target, key, sources, size, python, repo, env):
    from simapi import dataset
    clips, cosmos_dirs, val, seed = sources
    # On a failure the directory remains for inspection, without prepared.json.
    raw = target / 'raw'
    dataset.build(clips, str(raw), cosmos_dirs, val=val, seed=seed, size=size)
    split_path = raw / dataset.SPLIT_FILE
    validate_raw(raw, split_path)
    prepare(python, repo, env, raw, split_path, target / 'detector', synthetic=True)
    manifest = read_json(file(target / 'detector' / 'prepared.json'))
    manifest.update(variant=key, image_size=list(size))
    write_json(target / 'prepared.json', manifest)


def detector_env(repo, env):
    """Interpreter und Umgebung aus envs/detector; baut die CUDA-Ops des Repos."""
    python = uv_python(DETECTOR_ENV, '--locked')
    env = dict(env, PYTHONPATH=str(repo), FORCE_CUDA='1')
    # Upstream-Schritt `python setup.py develop`, aber nur die Erweiterungen im Repo: als Pakete
    # installiert wuerden sich models/, dataset/, ops/ der drei Repos gegenseitig ueberdecken.
    subprocess.run([*python, 'setup.py', 'build_ext', '--inplace'], cwd=repo, env=env, check=True)
    return python, env


def run(args):
    evaluate = args.mode == 'evaluate'
    config = Config(args.config)
    model = args.model
    settings = config.section(model)
    repo = config.existing(model, 'repo', folder=True)
    experiment = config.existing(model, 'experiment')
    backbone = config.existing(model, 'backbone_checkpoint')
    file(repo / 'dataset/nusc_mv_det_dataset.py', 'Detektor-Dataset')
    file(repo / 'evaluators/result2kitti.py', 'DAIR-Evaluator')
    model_class = settings.get('model_class')
    if not isinstance(model_class, str) or not model_class.isidentifier():
        raise ValueError(f'[{model}].model_class fehlt oder ist ungueltig')
    tree = ast.parse(experiment.read_text(encoding='utf-8'))
    if model_class not in {node.name for node in tree.body if isinstance(node, ast.ClassDef)}:
        raise ValueError(f'Modellklasse {model_class} fehlt in {experiment}')
    runtime = {**config.section('detector_runtime'), 'epochs': settings.get('epochs')}  # Epochen je Experiment
    gpus = gpu_ids(runtime.get('gpus'), 'detector_runtime')
    for key in ('batch_size', 'epochs'):
        if not isinstance(runtime.get(key), int) or isinstance(runtime[key], bool) or runtime[key] < 1:
            raise ValueError(f'{key} muss positiv sein ([detector_runtime].batch_size, [{model}].epochs)')
    if not isinstance(runtime.get('workers'), int) or runtime['workers'] < 0:
        raise ValueError('[detector_runtime].workers muss >= 0 sein')
    if runtime.get('precision') not in (16, 32):
        raise ValueError('[detector_runtime].precision: 16 oder 32 erforderlich')
    output = new_output(config.path_value('paths', 'runs') / ('evaluation' if evaluate else 'training') / safe_name(args.name))
    checkpoint = None
    if evaluate:
        checkpoint = str(file(Path(args.checkpoint).expanduser().absolute(), 'Evaluations-Checkpoint'))
        root = config.existing('dair', 'root', folder=True, nonempty=True)
        split_path = config.existing('dair', 'split')
        data_dir = config.path_value('dair', 'prepared')
        data = existing_dataset(data_dir)
        if data is None:
            validate_raw(root, split_path)
        elif data.get('synthetic') is not False:
            raise ValueError('Evaluation erfordert echten DAIR-Datensatz, keinen synthetischen Export')
        elif Path(data['raw_root']).resolve() != root or Path(data['split']).resolve() != split_path:
            raise ValueError('Vorbereiteter DAIR-Datensatz stimmt nicht mit [dair].root/[dair].split ueberein')
    else:
        key = dataset_key(args.data, args.cosmos, args.fullhd)
        size = FULLHD if args.fullhd else HD
        data_dir = config.path_value('paths', 'datasets') / key
        data = existing_dataset(data_dir)
        if data is None:
            sources = synthetic_sources(config, args.data, args.cosmos, size)
        elif data.get('synthetic') is not True or data.get('variant') != key:
            raise ValueError(f'{data_dir}/prepared.json passt nicht zur synthetischen Variante {key}')
        if args.resume:
            checkpoint = str(file(Path(args.resume).expanduser().absolute(), 'Resume-Checkpoint'))
    modules = ['numpy', 'torch', 'pytorch_lightning', 'mmcv', 'mmdet', 'mmdet3d', 'pyquaternion', 'ops.voxel_pooling']
    if data is None:
        for relative in CONVERTERS:
            file(repo / relative, 'Detektor-Konverter')
        modules += [relative[:-len('.py')].replace('/', '.') for relative in CONVERTERS]
    python, env = detector_env(repo, dict(offline_env(), CUDA_VISIBLE_DEVICES=','.join(map(str, gpus))))
    probe(python, modules, repo, env)
    print(f'Preflight OK: {model} {args.mode}, Datensatz {data_dir}' + (' (wird gebaut)' if data is None else '')
          + f', Ausgabe {output}')
    if args.check_only:
        return
    if data is None:
        if evaluate:
            prepare(python, repo, env, root, split_path, data_dir, synthetic=False)
        else:
            build_synthetic(data_dir, key, sources, size, python, repo, env)
        data = prepared_manifest(data_dir)
    if evaluate:
        data = dict(data, image_size=list(FULLHD))
    if not evaluate and data['counts']['train'] < runtime['batch_size'] * len(gpus):
        raise ValueError('Training hat weniger Bilder als eine vollstaendige globale Batch (drop_last=True)')
    # Training validiert auf den zurueckgehaltenen synthetischen Ansichten; DAIR wird getrennt evaluiert.
    request = {'action': args.mode, 'model': model, 'repo': str(repo), 'experiment': str(experiment),
               'model_class': model_class, 'backbone_checkpoint': str(backbone),
               'train_dataset': None if evaluate else data, 'val_dataset': data, 'checkpoint': checkpoint,
               'runtime': runtime, 'output': str(output)}
    execute([*python, '-u', SELF, '_detector', output / 'request.json'], repo, env, output, {'request.json': request})
    file(output / ('metrics.json' if evaluate else 'checkpoints/last.ckpt'), 'Ergebnis')


# --- Worker: laufen per `uv run --project envs/detector` in Python 3.8 mit NumPy 1.19 und Lightning 1.5

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_worker(request):
    """Info-Pickles in der NumPy-Umgebung des Detektors, damit keine NumPy-2-Pickles in NumPy 1.19 landen."""
    repo, root, output = Path(request['repo']), Path(request['raw_root']), Path(request['output'])
    sys.path.insert(0, str(repo))
    split = split_ids(request['split'])
    generator = load_module(repo / 'scripts/gen_info_dair.py', '_pipeline_dair_generator')
    # The official function reads data/single-infrastructure-split-data.json relative to the working
    # directory (mmcv.load or read_json, depending on the repo); give it exactly the requested split.
    with tempfile.TemporaryDirectory(prefix='dair-split-', dir=str(output)) as scratch:
        (Path(scratch) / 'data').mkdir()
        write_json(Path(scratch) / 'data' / 'single-infrastructure-split-data.json', split)
        os.chdir(scratch)
        try:
            for subset in ('train', 'val'):
                infos = generator.generate_info_dair(str(root), subset)
                if len(infos) != len(split[subset]):
                    raise ValueError(f'Info-Konverter lieferte unvollstaendigen {subset}-Split')
                with (output / (subset + '.pkl')).open('wb') as handle:
                    pickle.dump(infos, handle, protocol=4)
        finally:
            os.chdir(output)
    from scripts.data_converter.gen_kitti.label_json2kitti import (
        json2kitti,
        label_filter,
        rewrite_label,
    )
    from scripts.data_converter.gen_kitti.label_lidarcoord_to_cameracoord import (
        gen_lidar2cam,
    )
    labels = output / 'kitti_labels'
    with tempfile.TemporaryDirectory(prefix='dair-labels-', dir=str(output)) as tmp:
        gen_lidar2cam(str(root), tmp, label_type='camera')
        json2kitti(str(Path(tmp) / 'label/camera'), str(labels))
    rewrite_label(str(labels))
    label_filter(str(labels))
    for sid in split['train'] + split['val']:
        file(labels / (sid + '.txt'), 'Konvertiertes KITTI-Label', allow_empty=True)
    write_json(output / 'prepared.json', {
        'raw_root': str(root), 'split': str(Path(request['split']).resolve()),
        'train_info': str(output / 'train.pkl'), 'val_info': str(output / 'val.pkl'),
        'gt_labels': str(labels), 'synthetic': request['synthetic'],
        'generator_repo': str(repo), 'counts': {key: len(split[key]) for key in ('train', 'val')},
    })


class PipelineMixin:
    """Ersetzt die fest eingetragenen data/-Pfade der Originalexperimente durch die des Manifests."""

    def _loader(self, train):
        import dataset.nusc_mv_det_dataset as source
        import torch
        manifest = self._pipeline_request['train_dataset' if train else 'val_dataset']
        width, height = manifest['image_size']
        # Die Experimente tragen H = 1080, W = 1920 fest ein; das Netz sieht danach immer final_dim.
        kwargs = {'ida_aug_conf': dict(self.ida_aug_conf, H=height, W=width), 'classes': self.class_names,
                  'data_root': manifest['raw_root'], 'info_path': manifest['train_info' if train else 'val_info'],
                  'is_train': train, 'img_conf': self.img_conf, 'return_depth': False,
                  'use_cbgs': False, 'num_sweeps': self.num_sweeps,
                  'sweep_idxes': self.sweep_idxes, 'key_idxes': self.key_idxes}
        if 'is_bda' in inspect.signature(source.NuscMVDetDataset).parameters:
            kwargs['is_bda'] = train
        data = source.NuscMVDetDataset(**kwargs)
        if not len(data):
            raise ValueError('Leerer Detektor-Datensatz')
        return torch.utils.data.DataLoader(data, batch_size=self.batch_size_per_device,
                                           num_workers=self._pipeline_request['runtime']['workers'],
                                           shuffle=train, drop_last=train,
                                           collate_fn=partial(source.collate_fn, is_return_depth=False))

    def train_dataloader(self):
        return self._loader(True)

    def val_dataloader(self):
        return self._loader(False)

    def test_dataloader(self):
        return self._loader(False)


def make_evaluator(model, request):
    from evaluators import result2kitti as converters
    from evaluators.det_evaluators import RoadSideEvaluator
    from evaluators.kitti_utils import kitti_common
    from evaluators.kitti_utils.eval import kitti_eval

    # BEVHeight++ names its DAIR converter result2kitti_dair (its result2kitti is for KITTI)
    result2kitti = getattr(converters, 'result2kitti_dair', converters.result2kitti)
    # Predictions werden aufs Bild beschnitten, upstream fest auf 1920 x 1080
    converters.bbbox2bbox.__defaults__ = (list(request['val_dataset']['image_size']),)

    class PipelineEvaluator(RoadSideEvaluator):
        def evaluate(self, results, img_metas, **kwargs):
            # Never infer DAIR vs Rope3D from a substring of a user-provided path.
            target = Path(request['output'])
            if request['action'] == 'train':
                target = target / 'validation' / f'epoch_{model.current_epoch:03d}'
            target.mkdir(parents=True, exist_ok=True)
            self.output_dir = str(target)
            manifest = request['val_dataset']
            wanted = set(split_ids(manifest['split'])['val'])
            actual = [Path(meta['token']).stem for meta in img_metas]
            if len(actual) != len(wanted) or set(actual) != wanted:
                raise ValueError('Evaluator erhielt nicht genau den vollstaendigen Validierungssplit')
            formatted, _ = self.format_results(results, img_metas)
            pred_dir = result2kitti(formatted['img_bbox'], str(target), manifest['raw_root'], manifest['gt_labels'], demo=False)
            predicted = {p.stem for p in Path(pred_dir).glob('*.txt')}
            if predicted != wanted:
                raise ValueError('KITTI-Predictions unvollstaendig oder durch alte Dateien verunreinigt')
            ids = sorted(map(int, wanted))
            pred_annos = kitti_common.get_label_annos(pred_dir, image_ids=ids)
            gt_annos = kitti_common.get_label_annos(manifest['gt_labels'], image_ids=ids)
            text, metrics = kitti_eval(gt_annos, pred_annos, current_classes=['Car', 'Pedestrian', 'Cyclist'], metric='R40')
            (target / 'metrics.txt').write_text(text, encoding='utf-8')
            # numpy scalar metrics must be serializable.
            write_json(target / 'metrics.json', {'dataset': 'DAIR-V2X-I' if request['action'] == 'evaluate' else 'synthetic-val',
                                                'split': 'val', 'samples': len(ids), 'metric': 'KITTI R40',
                                                'values': {key: float(value) for key, value in metrics.items()}})
            print(text, flush=True)
            return metrics

    return PipelineEvaluator(class_names=model.class_names, current_classes=['Car', 'Pedestrian', 'Cyclist'],
                             data_root=request['val_dataset']['raw_root'],
                             gt_label_path=request['val_dataset']['gt_labels'], output_dir=request['output'])


def detector_worker(request):
    """Training/Evaluation mit den publizierten Lightning-1.x-Experimenten, ohne Repo-Dateien zu aendern."""
    sys.path.insert(0, request['repo'])
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    runtime = request['runtime']
    if not torch.cuda.is_available() or torch.cuda.device_count() < len(runtime['gpus']):
        raise RuntimeError('Konfigurierte CUDA-GPUs nicht verfuegbar')
    module = load_module(request['experiment'], '_pipeline_experiment')
    base = getattr(module, request['model_class'])
    # Keep a global class name so Lightning's DDP relaunch/pickling can resolve it.
    model_type = type('PipelineModel', (PipelineMixin, base), {'__module__': __name__})
    globals()['PipelineModel'] = model_type
    backbone = copy.deepcopy(module.backbone_conf)
    backbone['img_backbone_conf']['init_cfg'] = {'type': 'Pretrained', 'checkpoint': request['backbone_checkpoint']}
    # Rebind original global evaluator path before __init__, without editing repo files.
    module.gt_label_path = request['val_dataset']['gt_labels']
    pl.seed_everything(runtime.get('seed', 0), workers=True)
    output = Path(request['output'])
    # BEVHeight++ reads data/dair-v2x-i-kitti/ImageSets/val.txt relative to the working directory in __init__.
    val_list = output / 'data/dair-v2x-i-kitti/ImageSets/val.txt'
    val_list.parent.mkdir(parents=True, exist_ok=True)
    val_list.write_text('\n'.join(split_ids(request['val_dataset']['split'])['val']) + '\n', encoding='utf-8')
    os.chdir(output)
    model = model_type(gpus=len(runtime['gpus']), data_root=request['val_dataset']['raw_root'],
                       batch_size_per_device=runtime['batch_size'], default_root_dir=str(output),
                       backbone_conf=backbone, head_conf=copy.deepcopy(module.head_conf))
    model._pipeline_request = request
    model.evaluator = make_evaluator(model, request)
    train = request['action'] == 'train'
    callbacks = [ModelCheckpoint(dirpath=str(output / 'checkpoints'), filename='epoch{epoch:03d}',
                                 every_n_epochs=1, save_last=True, save_top_k=-1)] if train else []
    kwargs = {'gpus': len(runtime['gpus']), 'accelerator': 'gpu', 'max_epochs': runtime['epochs'],
              'precision': runtime['precision'], 'default_root_dir': str(output), 'callbacks': callbacks,
              'logger': CSVLogger(str(output), name='logs'), 'num_sanity_val_steps': 0,
              'gradient_clip_val': 5, 'limit_val_batches': 1.0, 'enable_checkpointing': train}
    if len(runtime['gpus']) > 1:
        kwargs['strategy'] = 'ddp'
    trainer = pl.Trainer(**kwargs)
    if train:
        trainer.fit(model, ckpt_path=request['checkpoint'])
    else:
        trainer.test(model, ckpt_path=request['checkpoint'])


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest='mode', required=True)
    train = modes.add_parser('train', help='auf einer synthetischen Variante trainieren, baut sie bei Bedarf')
    train.add_argument('--data', required=True, choices=('carla', 'cosmos'))
    train.add_argument('--cosmos', choices=('2.5', '3'), default='2.5', help='Cosmos-Version bei --data cosmos')
    train.add_argument('--fullhd', action='store_true', help='Trainingsbilder in 1920x1080 statt 1280x720')
    train.add_argument('--resume', help='lokaler Lightning-Checkpoint zum Fortsetzen')
    evaluate = modes.add_parser('evaluate', help='einen Checkpoint auf echtem DAIR-V2X-I val evaluieren')
    evaluate.add_argument('--checkpoint', required=True, help='lokale Lightning-.ckpt-Datei')
    for sub in (train, evaluate):
        sub.add_argument('--model', choices=MODELS, default='bevheight++')
        sub.add_argument('--name', required=True, help='eindeutiger Name fuer diesen Lauf')
        add_common_arguments(sub)
    for name in ('_prepare', '_detector'):
        modes.add_parser(name, help='intern, laeuft in der Detektor-Umgebung').add_argument('request')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse()
    if args.mode == '_prepare':
        prepare_worker(read_json(args.request))  # Fehler mit vollem Traceback ins run.log
    elif args.mode == '_detector':
        detector_worker(read_json(args.request))
    else:
        raise SystemExit(entrypoint(run, args))
