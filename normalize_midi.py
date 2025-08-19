from remi_z import MultiTrack
from sonata_utils import jpath, create_dir_if_not_exist

def main():
    flatten_midi('caihong.mid')
    # procedures()

def procedures():
    normalize_midi('caihong.mid')
    flatten_midi('caihong.mid')

def normalize_midi(midi_fn):
    midi_fp = jpath('misc', 'midis', midi_fn)
    out = jpath('misc', 'midi_normalized', midi_fn)
    mt = MultiTrack.from_midi(midi_fp)
    mt.set_tempo(90)
    mt.to_midi(out)

def flatten_midi(midi_fn):
    midi_fp = jpath('misc', 'midis', midi_fn)
    out_dir = jpath('misc', 'midi_flattened')
    create_dir_if_not_exist(out_dir)
    out_fp = jpath(out_dir, midi_fn)
    mt = MultiTrack.from_midi(midi_fp)
    mt.quantize_to_16th()
    mtf = mt.flatten()
    mtf.set_tempo(90)  # Set a default tempo if needed
    mtf.to_midi(out_fp)
    print(len(mtf.get_all_notes()))

if __name__ == "__main__":
    main()