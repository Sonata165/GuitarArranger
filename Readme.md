This tool generates solo guitar arrangements from any MIDI song. It models both left-hand fingering and right-hand picking patterns. The system consists of two main modules: **Voicer**, which finds economical fretboard positions to play melody and harmony simultaneously across all musical blocks; and **Arpeggiator**, which generates picking patterns that simulate the rhythmic feel of the original piece.

## Architecture

- **`rule_based.py`** — Main `ArrangerSystem` class. Pipeline: load MIDI → quantize → extract melody → extract chords → run Voicer → run Arpeggiator → output.
- **`fretboard_util.py`** — Models the guitar fretboard; maps pitches to `(string, fret)` positions.
- **`tab_util.py`** — `Chart`, `Tab`, `TabSeq` classes for representing and rendering guitar tablature.
- **`ga.py`** — Genetic algorithm approach (alternative to rule-based) for finding optimal fretboard positions.
- **`normalize_midi.py`** — Preprocesses input MIDIs.
- **`REMI-z/`** — Submodule for MIDI encoding/tokenization (`remi_z` package).
- **`SonataUtil/`** — Personal utility library.

## Output

For each processed segment (e.g. `caihong_bar_0_4`), the system writes the following files to `outputs/<song>/<segment>/`:

| File | Description |
|------|-------------|
| `<segment>.mid` / `.wav` | Full arrangement (melody + chord voicing) as MIDI and rendered audio |
| `<segment>_melody.mid` / `_melody.wav` | Isolated melody line as MIDI and rendered audio |
| `<segment>_chart.txt` | Per-block fretboard diagrams showing chord name, melody notes, and exact finger positions per string/fret (`X` = muted) |
| `<segment>_tab.txt` | Full 6-string guitar tablature across time (fret numbers or `--` for silence) |

## Usage

    python rule_based.py