This tool generates solo guitar arrangements from any MIDI song. It models both left-hand fingering and right-hand picking patterns. The system consists of two main modules: **Voicer**, which finds economical fretboard positions to play melody and harmony simultaneously across all musical blocks; and **Arpeggiator**, which generates picking patterns that simulate the rhythmic feel of the original piece.

## Architecture

- **`rule_based.py`** — Main `ArrangerSystem` class. Pipeline: load MIDI → quantize → extract melody → extract chords → run Voicer → run Arpeggiator → output.
- **`fretboard_util.py`** — Models the guitar fretboard; maps pitches to `(string, fret)` positions.
- **`tab_util.py`** — `Chart`, `Tab`, `TabSeq` classes for representing and rendering guitar tablature.
- **`ga.py`** — Genetic algorithm approach (alternative to rule-based) for finding optimal fretboard positions.
- **`normalize_midi.py`** — Preprocesses input MIDIs.
- **`REMI-z/`** — Submodule for MIDI encoding/tokenization (`remi_z` package).
- **`SonataUtil/`** — Personal utility library.

## Usage

    python rule_based.py