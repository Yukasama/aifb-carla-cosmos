# Cosmos → Training → DAIR-Evaluation

Zwei Skripte, zugeschnitten auf obi (`/share/Yacin`): [run_cosmos.py](run_cosmos.py) für Cosmos 2.5 oder 3 und [run_detector.py](run_detector.py) für Training und Evaluation der Objektdetektoren; Gemeinsames liegt in [common.py](common.py). Alle Pfade stehen in [pipeline.toml](pipeline.toml); `--config` wählt eine andere Datei. Objektdetektor ist per Default **BEVHeight++**, mit `--model cobev` bzw. `heightformer` die anderen.

## Einrichtung mit uv

Im Ordner `python_api` auf obi:

```sh
uv sync                                   # Launcher-Umgebung (dieses Projekt)
uv run run_cosmos.py 3 --check-only      # richtet beim ersten Aufruf auch die Modellumgebung ein
```

Jede Stufe läuft in ihrer eigenen uv-Umgebung, weil sich die Stacks ausschließen (Python 3.12/NumPy 2, Python 3.13/torch 2.10, Python 3.8/torch 1.9):

| Aufgabe | uv-Projekt | Python-Pakete |
|---|---|---|
| Launcher, Datensatz-Bau | `python_api/` | `pyproject.toml`, `uv.lock` |
| Cosmos Transfer 2.5 | `Cosmos/cosmos-transfer2.5` | dessen `uv.lock`, Extra aus `[cosmos25].uv` |
| Cosmos 3 | `Cosmos/cosmos-framework` | dessen `uv.lock`, Gruppe aus `[cosmos3].uv` |
| BEVHeight++, CoBEV, HeightFormer | `python_api/envs/detector` | torch 1.9.0+cu111, mmcv-full 1.4.0, mmdet 2.19.0, mmdet3d 0.18.1, Lightning 1.5.10, NumPy 1.19.5 |

Die Skripte rufen `uv run --project <projekt>` auf; uv installiert die gelockten Pakete beim ersten Lauf, auch bei `--check-only`. Die Cosmos-Lockfiles bleiben unverändert (`--frozen`). cosmos-framework verlangt genau uv 0.11.29 (`required-version` im Repo).

Für die Detektoren baut die erste Einrichtung mmdet3d 0.18.1 und die CUDA-Ops der Repos (`python setup.py build_ext --inplace`, entspricht dem `setup.py develop` der Anleitungen). Dafür braucht obi ein CUDA-11-Toolkit (`nvcc`, über `CUDA_HOME`); torch 1.9 lehnt ein nvcc 12 ab. Die vortrainierten ResNet-Backbones (`backbone_checkpoint`) müssen unter `~/.cache/torch/hub/checkpoints/` liegen, Download-URL steht in `pipeline.toml`.

## Offline und sofortige Fehler

Modelle werden nie heruntergeladen: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` gelten für alle Modellprozesse.

- **Transfer 2.5** läuft direkt über `examples/inference.py`. Die Registry holt jede Datei per `uvx hf download`; offline liefert das nur Dateien aus dem Hub-Cache `[cosmos25].models` (`models--org--name/snapshots/…`, auch unter `models/hub`), so wie ihn `hf download` ohne `--local-dir` anlegt. `uvx` muss im PATH sein.
- **Cosmos 3** nutzt das Sim-Transfer-Rezept des Frameworks (`.agents/skills/cosmos3-nano-sim-transfer`): `Cosmos3-Nano-Sim-Transfer` als Checkpoint, Text-Tokenizer aus `Cosmos3-Nano-tok`, VAE `Wan2.2_VAE.pth` aus `Wan2.2-VAE`, übergeben als Experiment-Overrides.

Jeder Lauf prüft zuerst Konfiguration, Repository-Dateien, Eingaben und Modelldateien, danach die Imports in der Modellumgebung. Fehlende Dateien, defekte Symlinks, fehlende Checkpoint-Shards, unvollständige Datensätze und überlappende Train/Val-Splits führen zu Exitcode 1, bevor ein Modelllauf startet. Ausgabeordner dürfen noch nicht existieren; für einen neuen Versuch `output` bzw. `--name` ändern. Nach einem Fehler bleibt der begonnene Ausgabeordner mit `run.log` zur Diagnose erhalten.

`--check-only` gibt es überall: Eingaben prüfen, Umgebung einrichten, Imports testen, ohne Ausgabe, Datensatz-Aufbau, Inferenz oder Training.

## 1. Transfer mit Cosmos 2.5 oder 3

```sh
uv run run_cosmos.py 2.5 --check-only
uv run run_cosmos.py 2.5
uv run run_cosmos.py 3 --check-only
uv run run_cosmos.py 3
```

Das Skript liest alle fertigen Clips direkt in `paths.clips`; die Holdout-Testdaten von `record_scenes.py` im Unterordner `holdout/` bleiben außen vor, und ein Clip mit `"holdout": true` in `clip.json` bricht ab. Alle fünf Kontrollvideos und das Padding-Manifest müssen vorhanden sein, die Videos in 1920 × 1080 (Default von `simapi.convert`). Alte Clips, etwa in 1280 × 720, zuvor mit `python -m simapi.convert <clips> --force --every 1` neu umwandeln; dafür sind Rohbilder nötig.

Transfer 2.5 nimmt `transfer2.5.json` des Clips; mehrere GPU-Einträge starten `torch.distributed.run`. Das Ergebnis hat die Größe der Eingabe (1920 × 1080). Guardrails bleiben an und müssen dann in `models` liegen; `disable_guardrails = true` schaltet sie ab.

Cosmos 3 baut je Clip eine Sim-Transfer-Anfrage nach dem Muster von `prepare_hf_sample.py`: die Defaults des Rezepts, eine Steuerung (`[cosmos3].control`, Default `edge`, aus dem RGB-Bild nach dem Skalieren berechnet), der ganze Clip als ein Fenster (`num_frames` = `max_frames` = `num_video_frames_per_chunk`), Prompt aus `prompt.json`, autoregressiv auf genau einer GPU. Das Ergebnis hat das 480p-Format des Modells (832 × 480 bei 16:9, volles Sichtfeld, bis 3 % gestreckt); `simapi.dataset` skaliert es auf die Trainingsgröße.

Ausgabe: `<cosmos25.output>/results/<clip>.mp4` bzw. `<cosmos3.output>/results/<clip>/vision.mp4`, die Anfragen unter `<cosmos3.output>/requests/`. Nach dem Prozessende muss für **jeden** Clip ein Video mit der erwarteten Framezahl und FPS vorliegen; ein Batchlauf mit einzelnen fehlgeschlagenen Samples gilt nicht als Erfolg.

## 2. Detektor trainieren

```sh
uv run run_detector.py train --data carla --name bh-carla
uv run run_detector.py train --data cosmos --cosmos 2.5 --name bh-c25
uv run run_detector.py train --data cosmos --cosmos 3 --name bh-c3
uv run run_detector.py train --data carla --fullhd --name bh-carla-fullhd

uv run run_detector.py train --model cobev --data carla --name cobev-carla
uv run run_detector.py train --model heightformer --data cosmos --cosmos 2.5 --name hf-c25
```

| `--model` | Experiment | Backbone | Epochen |
|---|---|---|---|
| `bevheight++` (Default) | `exps/dair-v2x/bev_height_plus_lss_r101_864_1536_256x256_102.py` | ResNet-101 | 60 |
| `cobev` | `exps/dair-v2x/cobev_r50_128x128.py` | ResNet-50 | 100 |
| `heightformer` | `exps/dair-v2x/height_former_r34_836_1536_256_256_102.py` | ResNet-34 | 50 |

Trainiert wird in 1280 × 720, mit `--fullhd` in 1920 × 1080. Die Experimente tragen 1920 × 1080 fest ein (`H`, `W` der Bildaugmentierung); der Adapter setzt sie auf die Bildgröße des jeweiligen Datensatzes, das Netz bekommt danach in beiden Fällen sein `final_dim`. Die Evaluation auf echtem DAIR läuft immer in 1920 × 1080.

Epochen stehen je Detektor in `pipeline.toml` (Default der Experimentdatei), Batchgröße, GPUs, Worker und Präzision in `[detector_runtime]`. Zum Fortsetzen `--resume /pfad/last.ckpt` und einen neuen `--name` verwenden. Modell, Loss, Lernrate und Scheduler kommen aus dem Originalexperiment; der Adapter setzt DataLoader, lokale Backbone-Gewichte, Evaluator und Checkpoint-/Log-Pfade. Training validiert auf den zurückgehaltenen **synthetischen** Ansichten, DAIR wählt keine Epochen aus.

Ergebnisse: `<paths.runs>/training/<name>/checkpoints/last.ckpt`, Epochen-Checkpoints, CSV-Logs sowie KITTI-R40-Validierung pro Epoche.

### Trainingsvariante

Fehlt die gewählte Variante unter `paths.datasets`, baut der erste Trainingslauf sie (mit dem DAIR-Konverter des gewählten Modells; das Format ist für alle drei gleich). Spätere Läufe, auch mit einem anderen Modell, verwenden sie wieder. Ordner `carla-hd`, `cosmos25-hd`, `cosmos3-hd` und mit `--fullhd` `carla-fullhd`, `cosmos25-fullhd`, `cosmos3-fullhd`:

```text
<variante>/
  raw/                  DAIR-Rohformat aus simapi.dataset
  detector/train.pkl    offizielles Info-Format, erzeugt in der Detektor-Umgebung
  detector/val.pkl
  detector/kitti_labels/
  detector/run.log
  prepared.json         verwendete Pfade, Variante und Split-Zahlen
```

`carla` verwendet die CARLA-RGB-Videos (`cosmos/rgb.mp4`, 1920 × 1080; kleinere Exporte werden abgelehnt), `cosmos` die Ergebnisse der ausgewählten Cosmos-Version. Bilder, Kalibrierung und 2D-Boxen werden auf die Trainingsgröße skaliert. Bilder derselben Kameraposition bleiben im selben Split. Holdout-Clips (`paths.clips/holdout/`) kommen in keine Variante; ein Clip mit `"holdout": true` direkt in `paths.clips` bricht den Aufbau ab. Padding wird nicht zum Trainingsbild. Eine nichtleere synthetische Validierung ist Pflicht; bei nur einem Clip kann kein getrennter Split entstehen und der Aufbau schlägt fehl. Ein abgebrochener Aufbau (Ordner ohne `prepared.json`) wird nicht wiederverwendet: prüfen, löschen, neu starten. Für ein verlässliches Experiment alle Varianten mit denselben Quellclips, demselben Seed und unveränderten Kamerapositionen erstellen.

`scripts/gen_info_dair.py` und die DAIR/KITTI-Konverter des Detektors laufen unverändert; ihre fest eingetragene Split-Datei `data/single-infrastructure-split-data.json` bekommen sie in einem eigenen Arbeitsordner mit genau dem angeforderten Split. Pickles entstehen in der Detektor-Umgebung, damit keine NumPy-2-Pickles in NumPy 1.19 landen.

## 3. Auf echtem DAIR evaluieren

```sh
uv run run_detector.py evaluate --checkpoint /share/Yacin/pipeline/runs/training/bh-carla/checkpoints/last.ckpt --name bh-carla-on-dair
uv run run_detector.py evaluate --model cobev \
  --checkpoint /share/Yacin/pipeline/runs/training/cobev-carla/checkpoints/last.ckpt --name cobev-carla-on-dair
```

Der Checkpoint muss zum Experiment des gewählten Modells passen. Evaluation lädt genau **einen** Checkpoint und ausschließlich den echten DAIR-Validierungssplit; synthetische Manifeste werden abgelehnt. KITTI-Predictions müssen genau alle erwarteten IDs enthalten, auch leere Dateien für Bilder ohne Objekte.

Fehlt `[dair].prepared`, bereitet die erste Evaluation es aus dem originalen Infrastructure-Rohdatenordner (`[dair].root`) vor: `data_info.json`, `image/`, `velodyne/`, `label/camera/` und JSON-Kalibrierungen; ein KITTI-Ordner allein genügt nicht. Split ist die offizielle Datei aus `BEVHeight++/data/`; auch der Wrapper `infrastructure_split` wird unterstützt. Es werden keine DAIR-Bilder kopiert und keine DAIR-Eingaben verändert.

Ergebnisse unter `<paths.runs>/evaluation/<name>/`: `results_nusc.json`, `data/*.txt`, `metrics.txt`, `metrics.json` mit KITTI-R40-Metriken für Car, Pedestrian und Cyclist sowie `request.json`, `command.json`, `run.log`. Die Prediction-/GT-Konvertierungen der Detektoren (bei BEVHeight++ `result2kitti_dair`) samt Confidence-Filter bleiben unverändert; der Adapter umgeht nur fest eingetragene Pfade und die DAIR/Rope3D-Erkennung am Ordnernamen.

## Prüfung und Grenzen

```sh
uv run python -m unittest discover -s tests -p test_pipeline.py -v
uv run python -m unittest discover -s tests -p test_exports.py -v
```

Die CPU-Tests prüfen Fehlerabbrüche, Offline-Umgebung, Python-3.8-Syntax von run_detector.py, Sim-Transfer-Anfragen, Eingabe-Splits, Varianten, Pfadübergaben und Prozessfehler. Ein vollständiger GPU-Lauf ist damit nicht nachgewiesen. Cosmos-Bilder vor dem Detektortraining auf geometrische Übereinstimmung mit den CARLA-Labels kontrollieren, etwa mit `python -m simapi.overlay <clip> --cosmos <cosmos-ausgabe>/results` (Labels ins Ergebnisvideo gezeichnet, siehe README).

Quellen: [Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework), [BEVHeight++](https://github.com/yanglei18/BEVHeight_Plus), [CoBEV](https://github.com/MasterHow/CoBEV), [HeightFormer](https://github.com/zhangzhang2024/HeightFormer). Geänderte Upstream-APIs können eine Adapteranpassung erfordern.
