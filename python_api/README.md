# simapi

`run_cosmos.py` (**Cosmos 2.5 / 3**) und `run_detector.py` (**Detektor-Training auf CARLA-/Cosmos-Varianten, Evaluation auf echtem DAIR**) sind in [PIPELINE.md](PIPELINE.md) beschrieben. Sämtliche Pfade auf obi stehen in `pipeline.toml`; Abhängigkeiten installiert uv, fehlende Gewichte oder Eingaben führen zum Fehler, ohne automatische Downloads.

CARLA-Steuerung für Sim Control: Verkehr, Roadside-Kamera, Bounding Boxes und Clips für Cosmos-Transfer2.5, Cosmos 3 und die 3D-Detektoren BEVHeight, HeightFormer und CoBEV in einem Prozess. Die Welt läuft synchron mit festem Tick (1/16 s); ein Tick ist ein Frame der Clips.

```sh
uv sync
.venv/bin/python -m simapi --hide-occluded --export   # alle Flags: --help
.venv/bin/python record_scenes.py                     # headless aufnehmen: 20000 Frames aus allen Kamerapositionen
.venv/bin/python -m simapi.convert _out               # Clips umwandeln (passiert sonst automatisch)
.venv/bin/python -m simapi.dataset _out --cosmos <ergebnisse> --out <ordner>   # Datensatz im DAIR-V2X-I-Format
.venv/bin/python tests/test_simapi.py                 # Selbsttest ohne laufenden Server, mit CARLA-Wheel
.venv/bin/python -m unittest discover -s tests -p test_exports.py -v  # ohne CARLA-Wheel
```

Sim Control startet das Package selbst, sobald CARLA erreichbar ist, zunächst ohne Verkehr und Aufnahme. „Erzeugen“ schickt die Einstellungen als JSON-Liste derselben Flags (eine Zeile pro Update) über stdin, dazu kommen die Kommandos `{"save_camera": "Name"}`, `{"goto_camera": "Name"}` und `{"set_fov": Grad}`. Getrennt wird mit EOF auf stdin: Das Package zerstört dann seine Actors und schaltet den Sync-Modus ab. Unter Windows wäre jedes Signal ein harter Abbruch, und CARLA bliebe im Sync-Modus eingefroren stehen.

Zurück kommt die Liste „Im Bild“: Mit `--status-file PFAD` schreibt das Package etwa fünfmal pro Sekunde alle Objekte im Kamerabild (bis `--max-distance`) als JSON in diese Datei, nach Distanz sortiert, mit Modell (`blueprint_id`), Lackfarbe, sichtbaren Pixeln und `occluded` (weniger als `--min-visible-pixels` sichtbar), dazu unter `camera.fov` das aktuelle Sichtfeld. Dafür läuft die Instanz-Segmentierung immer mit, auch ohne `--hide-occluded`.

## Kamera

Die Kamera nimmt die Spectator-Ansicht auf, standardmäßig mit 90° horizontalem Sichtfeld wie das CARLA-Fenster. Die Brennweite stellt Sim Control mit einem Schieberegler ein (10–100 mm Kleinbild-Äquivalent, 36 mm Sensorbreite: 18 mm = 90°); intern ist das das Kommando `{"set_fov": Grad}` (10–130°). Das CARLA-Fenster selbst bleibt bei 90°: Bei längerer Brennweite zeichnet das Package den Bildausschnitt als Rahmen ein (nicht während einer Aufnahme), bei kürzerer sieht die Kamera mehr als das Fenster. Das echte Kamerabild zeigt Sim Control als Vorschau: Mit `--preview-file PFAD` (zusammen mit `--status-file`) schreibt das Package im Takt der Statusdatei ein JPEG mit 640 px Breite, außerhalb einer Aufnahme von einer eigenen kleinen RGB-Kamera mit demselben Sichtfeld. Ein neues Sichtfeld startet die Sensoren neu und verwirft einen laufenden Clip wie eine Kamerabewegung. Jeder Clip hält Sichtfeld und Intrinsik in `clip.json` fest; daraus entstehen die Kalibrierung für die Detektoren und das Objektiv in `prompt.json` (Weitwinkel ab 54°, Normal ab 29°, sonst Tele). `--quality fullhd` (Default) wählt 1920 × 1080, `--quality hd` wählt 1280 × 720; auch `--resolution 1920x1080` bzw. `1280x720` geht. Die Einstellung gilt gemeinsam für RGB, Tiefe und Instanzsegmentierung. Boxen werden während einer Aufnahme nicht gezeichnet, da die Kamera sie mitfilmen würde. Die Bewegungsunschärfe der RGB-Kamera ist aus, damit Kanten pixelgenau auf Tiefe und Segmentierung liegen.

„Ansicht speichern“ legt die aktuelle Spectator-Pose mit dem Sichtfeld (`fov`, Grad) in `cameras.json` ab, ein Klick auf eine gespeicherte Position setzt den Spectator wieder dorthin und übernimmt ihr Sichtfeld; Einträge ohne `fov` haben 90°. `record_scenes.py` nimmt jede Position mit ihrem Sichtfeld auf. Einträge löschen geht von Hand:

```json
{
  "Kreuzung Nord": { "map": "Town10HD_Opt", "x": -45.2, "y": 21.7, "z": 8.0, "pitch": -25.0, "yaw": 90.0, "roll": 0.0, "fov": 73.74 }
}
```

## Clips für Cosmos und die Detektoren

Die Aufnahme ist auf zwei Abnehmer zugeschnitten: Cosmos (Transfer2.5 oder Cosmos 3) macht aus dem CARLA-Bild ein realistisches, und BEVHeight, HeightFormer und CoBEV (monokulare 3D-Detektoren für Roadside-Kameras) trainieren auf dem Ergebnis. Die Kontrollvideos sollen die Geometrie erhalten; ein generatives Modell garantiert das nicht. Vor dem Training müssen generierte Objekte und projizierte CARLA-Boxen visuell auf Übereinstimmung geprüft werden.

```
Aufnahme             →  Umwandlung (automatisch)  →  Cosmos 2.5 / 3     →  python -m simapi.dataset  →  dair2kitti.py, gen_info_dair.py, Training
<clip>/frames/          <clip>/cosmos/  <clip>/dair/    Ergebnisvideos         DAIR-V2X-I-Ordner
```

### Aufnahme

`--export` (in Sim Control „Clips aufnehmen“) nimmt Clips mit fester Kamera auf, `--clip-frames` Frames lang (Default 93 = 5,8 s bei 16 fps: ein Abschnitt bei Transfer2.5, und es passt in einen Abschnitt von Cosmos 3 mit bis zu 121 Frames). Ein Clip, der unterbrochen wird (Kamera bewegt, ein Sensorbild fehlt, Export aus), wird verworfen und der nächste beginnt sofort. Clips mit derselben Kamera bilden einen Take und heißen `<Startzeit>_000`, `_001`, …

Aufgezeichnet wird jeder Fahrzeug-Actor (einschließlich Fahrrädern und Motorrädern) und Fußgänger ab einem sichtbaren Pixel, mit 3D-Box, unabhängig von `--max-distance`, `--min-visible-pixels` und `--hide-occluded` (die gelten nur für Anzeige und Liste). Ein **Label** bekommt bei der Umwandlung nur, was höchstens 150 m von der Kamera entfernt ist (bis zur Boxmitte) und was die Kamera wirklich sieht: mindestens 10 eigene Pixel (auf 1920 × 1080 umgerechnet) und mindestens 5 % der Box sichtbar, also im Bild und nicht verdeckt (`simapi.convert --max-distance`, `--min-pixels`, `--min-share`). Weiter entfernte Objekte, Autos hinter Häusern, Lkw oder Baumkronen, von denen nur ein paar Pixel durchscheinen, und schmale Reste am Bildrand bekommen keins; `dair/meta.json` zählt sie unter `hidden`. Ein deutlich sichtbares Objekt ohne Label würden die Detektoren dagegen als Hintergrund lernen: Sichtbare Fahrzeug-/Personeninstanzen ganz ohne Actor (also ohne 3D-Box, etwa statische Kartenobjekte) lassen die Umwandlung fehlschlagen. Nur mit `simapi.convert --allow-unlabeled` darf diese Prüfung ausdrücklich übergangen werden. Das zufällige Spawnen garantiert keine bestimmte Anzahl sichtbarer Radfahrer pro Kamera.

```
<export-dir>/20261112-153012_000/
  clip.json                    Map, fps, Frames, Kamera (Intrinsik in Pixeln, Sichtfeld, Pose in UE-Koordinaten)
  frames/0000.json             Objekte: Klasse, 3D-Box (Mittelpunkt, halbe Maße, Rotation; Welt, UE), sichtbare Pixel
  frames/rgb/0000.png          1920 × 1080 oder 1280 × 720, verlustfrei
  frames/depth/0000.png        meter = 1000 * (R + 256*G + 65536*B) / (256**3 - 1)
  frames/instance/0000.png     R = Klasse (CARLA-Tag, z. B. 14 = Car), G + 256*B = Actor-ID (16 Bit)
  cosmos/                      Eingabe für Cosmos, siehe unten
  dair/                        Labels für die Detektoren, siehe unten
  convert.log
```

Ein laufender Clip liegt in `<name>.partial/` und wird erst nach dem letzten Frame umbenannt. Ein harter Abbruch (SIGKILL, abgestürzter Prozess) hinterlässt solche Ordner; die können weg. Unter Windows halten Virenscanner oder Editor frisch geschriebene Dateien manchmal kurz offen; das Umbenennen wird deshalb bis zu 10 s wiederholt. Rohbilder brauchen etwa 5 MB pro Frame, ein Clip mit 93 Frames also rund 500 MB; `--drop-frames` (in Sim Control „Rohbilder danach löschen“) löscht sie nach erfolgreicher Umwandlung, danach lässt sich der Clip aber nicht mehr neu umwandeln.

### Viele Clips headless: `record_scenes.py`

```sh
.venv/bin/python record_scenes.py                                      # 20000 Frames, alle Positionen aus cameras.json
.venv/bin/python record_scenes.py --quality hd                         # 20000 Frames in 1280x720
.venv/bin/python record_scenes.py --dry-run                            # Plan und Speicher prüfen, ohne CARLA
.venv/bin/python record_scenes.py --every 10                           # optional nur jeden 10. Frame für Detektoren aufbereiten
.venv/bin/python record_scenes.py --frames 93                          # Testlauf: ein Clip (+ ein Holdout-Clip)
.venv/bin/python record_scenes.py --frames 4000 --positions Museum "Kreuzung 1"
.venv/bin/python record_scenes.py --no-holdout                         # ohne Holdout-Testdaten
```

Startet CARLA ohne Fenster (`-RenderOffScreen`; Pfad und Einstellungen oben im Skript: `CARLA_EXES`, der erste vorhandene Pfad wird gestartet, Fahrzeuge, Fußgänger, Ports …). Aufgenommen wird nur in ganzen Clips zu 93 Frames, reihum verteilt: 20000 Frames auf den acht gespeicherten Positionen ergeben 216 Clips, **27 je Position (2511 Frames)**, insgesamt 20088. Restclips, die Cosmos mit Standbildern auffüllen müsste, entstehen so nicht. 93 Frames bleiben ein Testclip der ersten Position. Geht die Clipzahl nicht auf, bekommen die ersten Positionen einen Clip mehr. Geparkte Autos der Karte (Layer `ParkedVehicles`) blendet das Skript aus, weil sie keine Actors sind und kein 3D-Label bekommen können. Fußgänger, die der Verkehr anfährt, löscht CARLAs Fußgänger-Navigation; das Package ersetzt sie alle 20 Ticks an Orten außerhalb des Kamerabilds, damit ihre Zahl über den ganzen Lauf gleich bleibt. Die acht Einträge in `cameras.json` sind Kamerapositionen; das Skript prüft nicht, ob sie acht räumlich verschiedene Kreuzungen zeigen.

**Holdout:** Die Position `Holdout` in `cameras.json` (andere mit `--holdout NAME`) gehört nie zum Training und zählt nicht zu `--frames`. Sie wird zusätzlich und zuletzt aufgenommen, mit so vielen Clips wie die größte Trainingsposition: bei 20000 Frames ebenfalls 27 Clips (2511 Frames), insgesamt also 22599. Ihre Clips landen in `<export-dir>/holdout/` und tragen `"holdout": true` in `clip.json`; umgewandelt werden sie wie alle anderen (`cosmos/`, `dair/`), damit sie später als Testdatensatz bereitstehen. Cosmos-Inferenz und Trainingsdatensatz bekommen sie nicht: `run_cosmos.py`, `run_detector.py` und `simapi.dataset` lesen nur Clips direkt im Export-Ordner, und ein Clip mit `"holdout": true` bricht dort mit Fehler ab. In `--positions` darf die Holdout-Position nicht stehen; fehlt sie in `cameras.json`, bricht das Skript ab (`--no-holdout` nimmt ohne sie auf, ins Training kommt sie auch dann nicht). `recording-<zeit>.json` führt sie getrennt unter `holdout`, die Zahlen oben beziehen sich nur aufs Training.

Standardmäßig wird jeder Frame für die Detektoren aufbereitet, also 20000 Trainingsbilder vor dem Train/Val-Split. `--every 10` reduziert diese Zahl ausdrücklich. Vor dem Start erscheinen Plan, geschätzter Speicherbedarf und Dauer; bei zu wenig Platz startet die Aufnahme nicht. Full HD mit allen Labels braucht grob 165 GB mit Rohbildern oder 60 GB mit `--drop-frames`, HD entsprechend weniger. Punktwolken pro Frame erhöhen den Bedarf erheblich. Höchstens zwei Umwandlungen laufen gleichzeitig; bei acht ausstehenden Clips wartet die Aufnahme. Am Ende schreibt `recording-<zeit>.json` Soll/Ist-Zahlen, Clipnamen pro Position, Auflösungen, Fehler und `complete`. Erfolg (Exitcode 0) gibt es nur bei erfülltem Frame-Budget und erfolgreichen Umwandlungen. Die alte Zeitmessung von etwa 3–4 fps und 25–30 s Umwandlung pro Clip bezog sich auf nur jeden zehnten Detektorframe; mit allen Labels ist sie nur eine grobe Startschätzung.

Der erste Clip jeder Position wird geprüft: Sieht die Tiefen- oder Segmentierungskamera nur ein Objekt direkt vor der Linse (die Kamera steckt in einem Mesh, das im Kamerabild unsichtbar ist) oder keine Straße, wird die Position übersprungen und am Ende mit Grund aufgeführt; dann die Position in Sim Control etwas verschieben und neu speichern. Läuft unter `--host`/`--port` schon ein CARLA, bricht das Skript ab, damit nicht versehentlich eins mit Fenster aufnimmt; mit `--use-running` wird es benutzt (eine Simulation, die es gerade steuert, vorher trennen). Ctrl+C bricht sauber ab, mit `--stop-on-eof` auch EOF auf stdin: fertige Clips bleiben, laufende Umwandlungen werden abgewartet, die übrigen lassen sich mit `python -m simapi.convert` nachholen. Ein selbst gestartetes CARLA wird am Ende beendet.

In Sim Control startet der Kasten „Testlauf“ das Skript mit `--frames 93 --use-running` über die eingestellte CARLA-Adresse: ein Clip der ersten Position und einer der Holdout-Position, aufgenommen und umgewandelt. Die Simulation wird dafür getrennt und verbindet sich danach wieder; die Ausgabe erscheint im Kasten, „Abbrechen“ schließt stdin.

### Umwandlung

Ist ein Clip fertig, startet die Simulation im Hintergrund `python -m simapi.convert <clip>` (Ausgabe in `convert.log`; läuft auch weiter, wenn die Simulation endet). Von Hand geht es genauso, mit einem Clip- oder dem Export-Ordner; schon umgewandelte Teile werden übersprungen (`--force` baut neu, `--help` zeigt die Optionen).

#### `cosmos/`: Eingabe für Cosmos-Transfer2.5 und Cosmos 3

Fünf Videos, standardmäßig 1920 × 1080 (`--size`), 16 fps, Frame für Frame deckungsgleich. Die Rohaufnahmen bleiben in der gewählten Aufnahmeauflösung. Beide Modelle lesen dieselben Kontrollvideos und skalieren selbst auf ihre Arbeitsgröße, ohne zu beschneiden (Transfer2.5 1280 × 704, Cosmos-3-Sim-Transfer 832 × 480); Cosmos 3 nutzt den Modell-Tier `720` bei 16:9. Ein nativer 1080p-Video-Tier wird dort nicht angeboten ([offizielle Auflösungstabelle](https://nvidia.github.io/cosmos-framework/inference/#resolution-tiers)):

| Datei | Inhalt | wie bei Cosmos |
|---|---|---|
| `rgb.mp4` | Kamerabild (Transfer2.5: `video_path`; Cosmos 3 braucht es nicht) | – |
| `depth.mp4` | inverse Tiefe, über den ganzen Clip auf 0–255 normiert, nah = hell, grau | wie Video Depth Anything in Transfer2.5 |
| `seg.mp4` | eine feste Farbe pro Segment, unbeschriftet schwarz; Fahrzeuge, Personen, Masten, Ampeln und Schilder einzeln, alles andere pro Klasse | Farben verteilt wie Transfer2.5 (SAM2-Masken) |
| `edge.mp4` | Canny-Kanten des Kamerabilds, Schwellen 100/200 | Preset `medium` |
| `vis.mp4` | Kamerabild verkleinert, Bilateralfilter (30/150/100), grob herunter- und hochskaliert; heißt bei Cosmos 3 `blur` | Preset `medium` |

Dazu je Modell eine Spec, Pfade relativ zur Datei, `name` = Clipname. Die Formate unterscheiden sich, und Cosmos 3 lehnt unbekannte Felder ab, deshalb zwei Dateien:

| | Transfer2.5 | Cosmos 3 |
|---|---|---|
| Spec | `transfer2.5.json` (`video_path`, edge 1.0, depth 0.5, seg 1.0 wie das Multicontrol-Beispiel von Transfer2.5, aber ohne `vis`: das gibt Farben und Licht des CARLA-Bilds vor und hält das Ergebnis nach Simulation aussehend; Prompt als Text, `--prompt`) | `cosmos3.json` (`model_mode` `video2video`, `resolution` 720, `aspect_ratio` 16,9, `num_frames` und `fps` wie der Clip, `blur`, `weight`; übrige Einstellungen wie in den Transfer-Beispielen von cosmos-framework) |
| Prompt | Text in der Spec | `prompt.json`: strukturierte Beschreibung im Schema der Cosmos-3-Beispiele (Objekte, Hintergrund, Licht, statische Kamera von oben …) mit der Zahl der sichtbaren (beschrifteten) Fahrzeuge, Zweiräder und Personen des Clips; negativer Prompt: der mitgelieferte von Cosmos 3 |
| Aufruf | `python examples/inference.py -i <export-dir>/*/cosmos/transfer2.5.json -o <ergebnisse>` | `python -m cosmos_framework.scripts.inference -i <export-dir>/*/cosmos/cosmos3.json -o <ergebnisse> --checkpoint-path Cosmos3-Nano` |
| Ergebnis | `<ergebnisse>/<clipname>.mp4` | `<ergebnisse>/<clipname>/vision.mp4` |

Transfer2.5 liefert das Ergebnis in der Größe der Eingabe (1920 × 1080), Cosmos-3-Sim-Transfer in 832 × 480. `manifest.json` dokumentiert Originalframes, Videoframes, Padding, FPS und Videogröße. Restclips werden durch Wiederholung des letzten Frames auf mindestens 93 und eine Länge von `4k+1` aufgefüllt: der 82-Frame-Rest hat 11 zusätzliche Videoframes. `cosmos3.json` und `prompt.json` verwenden die aufgefüllte Länge. Der Dataset-Builder übernimmt ausschließlich die original beschrifteten Frames; Padding wird nie zum Trainingsbild. Edge und Blur sind aus dem Quellcode von Cosmos-Transfer2.5 (`datasets/augmentors/control_input.py`) nachgebaut und werden in Videoauflösung berechnet. Kodiert wird verlustfrei (H.264, QP 0, yuv444p, Preset `veryfast`). Für die NIM gelten separate API-Schemas und Uploadlimits; die exportierten Specs richten sich an die Python-Inferenzskripte der Repositories.

#### `dair/`: Labels für BEVHeight, HeightFormer und CoBEV

Adressiert sind [Roadside-HeightFormer von Zhang et al.](https://github.com/zhangzhang2024/HeightFormer) und [CoBEV](https://github.com/MasterHow/CoBEV) mit dem DAIR-V2X-I-Datenpfad. Andere gleichnamige HeightFormer-Modelle können andere Datensätze benötigen. `dair/` enthält die Rohdaten für deren Loader, standardmäßig für **jeden Frame** (`--every 1`) und bereits für die DAIR-Bildgröße kalibriert (1920 × 1080, in Skripten und Configs fest eingetragen):

```
dair/meta.json                         Frames, Kamerahöhe, Bodenebene, Schwellen für ein Label, Objekte ohne Label (hidden) und unbeschriftete Objekte
dair/calib/camera_intrinsic.json       cam_K (Pixelmitten-Konvention), cam_D = 0
dair/calib/virtuallidar_to_camera.json rotation, translation: Bodensystem → Kamera (OpenCV-Achsen)
dair/label/0000.json                   wie DAIR label/camera
dair/velodyne/0000.pcd                 Punktwolke aus dem Tiefenbild, x y z intensity (float32, binär)
```

Bezugssystem ist bei DAIR ein „virtuelles LiDAR“, dessen Ebene z = 0 die Fahrbahn ist; daraus leiten BEVHeight und CoBEV die Höhe über Grund ab (`denorm`). Hier liegt der Ursprung auf der Fahrbahn senkrecht unter der Kamera, z zeigt senkrecht zur Fahrbahn nach oben, x in Blickrichtung, y nach links. Die Fahrbahnebene wird aus den Straßenpixeln von Tiefen- und Segmentierungsbild geschätzt (`camera_height` in `meta.json` zur Kontrolle). Die Detektoren sehen 0 bis 102,4 m nach vorn und ±51,2 m zur Seite (Configs `*_102`).

Ein Label pro sichtbarem Objekt bis `--max-distance` (Default 150 m, Kamera bis Boxmitte; 0 = unbegrenzt). Sichtbar heißt: mindestens `--min-pixels` eigene Pixel (Default 10, auf 1920 × 1080 umgerechnet; nur Fahrzeug-/Personenklassen zählen) und ein sichtbarer Anteil von mindestens `--min-share` (Default 0,05). Der Anteil ist das Produkt aus dem Teil der projizierten 3D-Box, der im Bild liegt, und dem unverdeckten Teil darin (sichtbare Pixel / (sichtbare + verdeckende); verdeckend ist ein fremdes Pixel in der Box, das laut Tiefenbild mehr als 0,5 m näher ist als die nächste Boxecke). Laub mit Lücken zählt als Verdeckung; ein Auto, von dem weniger als 5 % durch die Blätter blitzen, bekommt kein Label. Dieselbe Regel bestimmt die Objektzahlen in `prompt.json`.

| Feld | Inhalt |
|---|---|
| `type` | `Car`, `Van`, `Truck`, `Bus`, `Motorcyclist`, `Cyclist`, `Pedestrian` (aus dem CARLA-`base_type`; ohne: `Car`). BEVHeight fasst beim Umwandeln Van/Truck/Bus zu Car und Motorcyclist zu Cyclist zusammen |
| `3d_location`, `3d_dimensions`, `rotation` | Mittelpunkt, h/w/l in m, Gierwinkel um z (0 = entlang x, gegen den Uhrzeigersinn), im Bodensystem |
| `2d_box` | eng um die sichtbaren Pixel |
| `occluded_state` | 0 frei (≥ 90 % unverdeckt), 1 teilweise (≥ 50 %), 2 überwiegend verdeckt (< 50 %); aus Tiefenbild und projizierter Box geschätzt |
| `truncated_state` | 0 im Bild, 1 links/rechts, 2 oben/unten abgeschnitten |
| `alpha` | Beobachtungswinkel wie KITTI (BEVHeight rechnet ihn ohnehin neu) |

`occluded_state` und `truncated_state` steuern nur die Schwierigkeitsstufen der Auswertung. Fahrzeuge oder Personen im Bild ohne Actor und damit ohne 3D-Box (etwa geparkte Autos, die zur Karte gehören statt Actors zu sein; `record_scenes.py` blendet sie aus) meldet `convert.log`, und `meta.json` zählt sie pro Frame unter `unlabeled`; die absichtlich ausgelassenen (zu weit weg oder zu wenig sichtbar) stehen getrennt unter `hidden`.

Ohne `--allow-unlabeled` scheitert die Umwandlung bereits ab einer sichtbaren Instanz ohne Actor. Andere Schwellen gelten nach `python -m simapi.convert <ordner> --force --min-share 0.1` (Rohbilder nötig). Die Bodenebene ist eine Schätzung aus Straßenpixeln; bei mehreren Fahrbahnhöhen oder stark gekrümmter Straße ist die Kalibrierung visuell zu kontrollieren. Die exportierten Detektorboxen verwenden aufrechte Boxen und einen Gierwinkel wie DAIR.

### Datensatz bauen

```sh
.venv/bin/python -m simapi.dataset _out --cosmos <ergebnisse> --out <BEVHeight>/data/dair-v2x-i
.venv/bin/python -m simapi.dataset _out --cosmos <transfer25> --cosmos <cosmos3> --out <ordner>   # beide Ergebnisse
.venv/bin/python -m simapi.dataset _out --out <ordner>        # ohne Cosmos: CARLA-Bilder
.venv/bin/python -m simapi.dataset _out --out <ordner> --fullhd   # Bilder in 1920 × 1080 statt 1280 × 720
```

Die CARLA-Bilder kommen aus `cosmos/rgb.mp4` (1920 × 1080, verlustfrei). Ältere Exporte in 1280 × 720 mit `python -m simapi.convert <ordner> --force` neu umwandeln (Rohbilder nötig).

`--cosmos` erkennt beide Ablagen (`<clipname>.mp4` und `<clipname>/vision.mp4`) und geht mehrfach: Jeder Ausgabeordner (zwei Modelle, verschiedene Prompts) gibt je Clip eigene Bilder mit denselben Labels, und alle Bilder eines Clips landen zusammen im Training oder in der Validierung. Nimmt alle umgewandelten Clips (mit `dair/`), deren Video vorhanden ist, skaliert die Bilder auf 1280 × 720 (mit `--fullhd` 1920 × 1080), rechnet Kalibrierung und 2D-Boxen von `dair/` auf diese Größe um und schreibt das Rohformat von DAIR-V2X-I mit fortlaufenden IDs: `image/`, `velodyne/`, `calib/camera_intrinsic/`, `calib/virtuallidar_to_camera/`, `label/camera/` (und gleich `label/virtuallidar/`), `data_info.json` (mit Herkunft: Clip, Frame, Ausgabeordner oder `carla`) und `single-infrastructure-split-data.json`. Aufgeteilt wird nach Kameraposition (`--val`, Default 20 %), damit Training und Validierung nie dieselbe Ansicht zeigen; bei nur einer Position nach Clips. Der Zielordner muss neu oder leer sein.

Im Repo des Detektors (BEVHeight oder CoBEV) geht es weiter wie bei DAIR-V2X-I: die Split-Datei nach `data/` kopieren, dann

```sh
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
python scripts/data_converter/visual_tools.py --data_root data/dair-v2x-i-kitti --demo_dir ./demo   # Boxen prüfen
python scripts/gen_info_dair.py
```

`dair2kitti.py` liest die Punktwolken mit `pypcd`; das Training selbst nutzt sie nicht (`return_depth=False`). Die Experimente setzen `H = 1080`, `W = 1920` voraus: für einen Datensatz in 1280 × 720 dort `H = 720`, `W = 1280` eintragen (`run_detector.py` macht das selbst).

`simapi.dataset` prüft FPS, vollständige Videolänge inklusive dokumentiertem Padding und identisches Seitenverhältnis. Fehlende Clips, angeforderte Cosmos-Ergebnisse oder beschriftete Frames führen zu einem Fehler. Eine Inhaltsprüfung generierter Objekte ersetzt das nicht. Bei vorhandenen alten Exporten `python -m simapi.convert <ordner> --force --every 1` ausführen, um vollständige Labels und das Padding-Manifest neu zu erstellen (Rohbilder erforderlich).

Bei den Detektoren sind zusätzlich deren lokale Pfade einzustellen: HeightFormer erwartet im offiziellen `scripts/gen_info_dair.py` die Rohdaten unter `./data/dair-v2x-i`. CoBEV hat dort einen absoluten Autorenpfad eingetragen; `dair_root` auf den eigenen **DAIR-Rohdatenordner** mit `image/`, `label/camera/` und den JSON-Kalibrierungen ändern und die erzeugten Pickle-Dateien an den in der Trainingsconfig erwarteten Ort legen. Eine erfolgreiche CARLA-/Cosmos-/Trainingsausführung auf der Ziel-GPU ist durch die lokalen Exporttests nicht nachgewiesen.

Das `carla`-Wheel 0.9.16 gibt es nur für Linux (x86_64) und Windows; die Version sollte zum CARLA-Server passen. Auf anderen Systemen wird es nicht installiert: Umwandlung, Datensatz, die Tests ohne CARLA und `record_scenes.py --dry-run` laufen dort trotzdem.
