#!/usr/bin/env bash
# Startet Cosmos Transfer 2.5 fuer einen Clip-Ordner, der neben diesem Skript liegt.
#
# Aufbau im Home-Verzeichnis (Beispiel):
#   ~/run_cosmos.sh
#   ~/j00_clip0_inv/  rgb.mp4 edge.mp4 depth.mp4 seg.mp4 spec_*.json
#
# Aufruf:
#   bash ~/run_cosmos.sh j00_clip0_inv                  # alle drei Kontrollen
#   bash ~/run_cosmos.sh j00_clip0_inv edge_depth       # nur edge + depth
#   GPU=1 bash ~/run_cosmos.sh j00_clip0_inv            # auf GPU 1
#
# Die Pfade in den Specs (z. B. Windows-Pfade) werden automatisch auf den
# Linux-Pfad des Clip-Ordners umgeschrieben. Die Originale bleiben unveraendert.
# Nur fuer die alten Beispielclips. Die neue Pipeline ist python_api/run_cosmos.py.
# Fehlende Umgebungen, Modelle oder Caches fuehren sofort zu einem Fehler.
set -euo pipefail

# ---------- Einstellungen ----------
VENV="${VENV:-$HOME/venv}"                                  # vorhandenes Python-venv
COSMOS_DIR="${COSMOS_DIR:-/share/Yacin/Cosmos/cosmos-transfer2.5}"
GPU="${GPU:-1}"                                             # freie GPU laut nvidia-smi
MODELS_ENV="$COSMOS_DIR/models/env.sh"                      # Yacins Setup: Gewichte aus diesem Ordner, offline
# -----------------------------------

CLIP_NAME="${1:?Bitte Clip-Ordner angeben, z. B.: bash run_cosmos.sh j00_clip0_inv}"
CONFIG="${2:-edge_depth_seg}"                               # edge | edge_depth | edge_depth_seg

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CLIP_DIR="$SCRIPT_DIR/$CLIP_NAME"
SPEC_IN="$CLIP_DIR/spec_${CONFIG}.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$CLIP_DIR/out_${CONFIG}_${STAMP}"
SPEC_LINUX="$OUT_DIR/spec_${CONFIG}.linux.json"
LOG="$OUT_DIR/run.log"

# ---------- Pruefungen ----------
[[ -d "$CLIP_DIR" ]]   || { echo "Clip-Ordner fehlt: $CLIP_DIR"; exit 1; }
[[ -f "$SPEC_IN" ]]    || { echo "Spec fehlt: $SPEC_IN"; exit 1; }
for v in rgb edge; do
  [[ -f "$CLIP_DIR/$v.mp4" ]] || { echo "Video fehlt: $CLIP_DIR/$v.mp4"; exit 1; }
done
[[ "$CONFIG" == edge_depth* ]] && { [[ -f "$CLIP_DIR/depth.mp4" ]] || { echo "depth.mp4 fehlt"; exit 1; }; }
[[ "$CONFIG" == edge_depth_seg ]] && { [[ -f "$CLIP_DIR/seg.mp4" ]] || { echo "seg.mp4 fehlt"; exit 1; }; }
[[ -f "$COSMOS_DIR/examples/inference.py" ]] || { echo "Cosmos nicht gefunden: $COSMOS_DIR"; exit 1; }
[[ -f "$MODELS_ENV" ]] || { echo "Gewichte-Setup nicht gefunden: $MODELS_ENV"; exit 1; }

# ---------- vorhandene Laufzeit pruefen ----------
export PATH="$HOME/.local/bin:$PATH"   # Cosmos ruft uvx auf, um Checkpoints zu finden
command -v uvx >/dev/null || { echo "uvx fehlt; keine automatische Installation"; exit 1; }
[[ -x "$VENV/bin/python" ]] || { echo "Python-venv fehlt: $VENV"; exit 1; }
[[ -f "$VENV/bin/activate" ]] || { echo "venv-Aktivierung fehlt: $VENV/bin/activate"; exit 1; }

# ---------- Umgebung ----------
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Karte per UUID festlegen: genau die, die nvidia-smi als Nummer $GPU zeigt (CUDA zaehlt sonst evtl. anders)
GPU_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)" || { echo "GPU $GPU laut nvidia-smi nicht gefunden"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
# shellcheck disable=SC1090
source "$MODELS_ENV"                   # setzt HF_HUB_CACHE, COSMOS_CACHE_DIR, HF_HUB_OFFLINE=1
[[ -n "${HF_HUB_CACHE:-}" && -d "$HF_HUB_CACHE" ]] || { echo "HF_HUB_CACHE fehlt oder existiert nicht"; exit 1; }
[[ -n "$(ls -A "$HF_HUB_CACHE")" ]] || { echo "HF_HUB_CACHE ist leer: $HF_HUB_CACHE"; exit 1; }
[[ -n "${COSMOS_CACHE_DIR:-}" && -d "$COSMOS_CACHE_DIR" ]] || { echo "COSMOS_CACHE_DIR fehlt oder existiert nicht"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT_DIR"

# ---------- Spec auf Linux-Pfade umschreiben ----------
python - "$SPEC_IN" "$SPEC_LINUX" "$CLIP_DIR" <<'PY'
import json, os, sys
src, dst, clip = sys.argv[1:4]
s = json.load(open(src, encoding="utf-8"))
s["video_path"] = os.path.join(clip, "rgb.mp4")
for k in ("edge", "depth", "seg"):
    if k in s and "control_path" in s[k]:
        s[k]["control_path"] = os.path.join(clip, k + ".mp4")
json.dump(s, open(dst, "w", encoding="utf-8"), indent=2)
PY

# ---------- Info ----------
{
  echo "Start:        $(date)"
  echo "Host:         $(hostname)"
  echo "GPU:          $GPU ($GPU_UUID)"
  nvidia-smi -i "$GPU" --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv,noheader || true
  echo "Clip:         $CLIP_DIR"
  echo "Konfiguration:$CONFIG"
  echo "Spec:         $SPEC_LINUX"
  echo "Ausgabe:      $OUT_DIR"
  echo "----------------------------------------"
  cat "$SPEC_LINUX"
  echo "----------------------------------------"
} | tee "$LOG"

# ---------- Inferenz ----------
cd "$COSMOS_DIR"
START=$(date +%s)
python examples/inference.py -i "$SPEC_LINUX" -o "$OUT_DIR" 2>&1 | tee -a "$LOG"
END=$(date +%s)

echo "Fertig nach $(( (END-START)/60 )) min $(( (END-START)%60 )) s. Ergebnis in: $OUT_DIR" | tee -a "$LOG"
ls -lh "$OUT_DIR" | tee -a "$LOG"
