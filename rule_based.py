"""
这个文件用来描述Rule-Based Guitar Arranger的伪代码
"""

from remi_z import MultiTrack, NoteSeq, ChordSeq, note_name_to_midi_pitch
from typing import List
from voicer import Voicer
from arpeggiator import Arpeggiator
import subprocess
from pydub import AudioSegment
from sonata_utils import jpath, create_dir_if_not_exist


def main():
    arranger = ArrangerSystem()
    
    # midi_fp = "misc/001.mid"
    midi_fp = 'misc/midis/canon_in_D.mid'

    arranger.arrange_song_from_midi(
        midi_fp,
        start_bar=8,
        n_bars=8,
        # melody_method="inst_id_13",
        melody_method="hi_note_dur",
        pitch_shift=-9,
    )


class ArrangerSystem:
    def __init__(self):
        self.mt: MultiTrack = None
        self.voicer = Voicer()
        self.arpeggiator = Arpeggiator()

    def arrange_song_from_midi(
        self,
        midi_fp,
        output_dir=None,
        start_bar=0,
        n_bars=8,
        melody_method="hi_track_song",
        pitch_shift=0,
    ):
        song_name = midi_fp.split("/")[-1].split(".")[0]
        print(f"Arranging song: {song_name}")

        bar_end = start_bar + n_bars

        save_name = f"{song_name}_bar_{start_bar}_{bar_end}"
        save_dir = (
            output_dir
            if output_dir is not None
            else jpath("outputs", song_name, save_name)
        )
        create_dir_if_not_exist(save_dir)

        self.mt = MultiTrack.from_midi(midi_fp)[start_bar:bar_end]

        self.mt.quantize_to_16th()

        if pitch_shift != 0:
            self.mt.shift_pitch(pitch_shift)

        # Prepare model inputs
        notes_of_bars = self.get_all_notes()
        melody = self.extract_melody(melody_method)
        chord_of_bars = self.extract_chord()

        # Melody range normalization
        melody = self.normalize_melody_range(melody)

        # Save melody to file
        melody_fp = jpath(save_dir, f"{save_name}_melody.mid")
        melody_mt = MultiTrack.from_note_seqs(
            melody, program_id=25
        )  # 24在garadgeband里全是滑音，太奇怪了
        print(f"Saving melody to {melody_fp}")
        melody_mt.to_midi(melody_fp)

        # Synthesize the melody to WAV
        sf_path = "resources/Tyros Nylon.sf2"
        melody_audio_fp = jpath(save_dir, f"{save_name}_melody.wav")
        print(f"Synthesizing melody MIDI to WAV: {melody_audio_fp}")
        self.midi_to_wav(melody_fp, sf_path, melody_audio_fp)
        self.post_process_wav(
            melody_audio_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1
        )

        # Get chart sequence (left hand modeling)
        chart_seq = self.voicer.generate_chart_sequence_for_song(melody, chord_of_bars)
        if not chart_seq or len(chart_seq) == 0:
            raise ValueError(
                "No chart sequence generated. Please check model robustness."
            )
        chart_fp = jpath(save_dir, f"{save_name}_chart.txt")
        print(f"Saving chart sequence to {chart_fp}")
        self.voicer.save_chart_sequence_to_file(chart_seq, chart_fp)

        # Tab generation given charts (right hand modeling)
        song_tab = self.arpeggiator.arpeggiate_a_song(melody, chart_seq, notes_of_bars)

        # Save the tab to file
        tab_fp = jpath(save_dir, f"{save_name}_tab.txt")
        print(f"Saving tab to {tab_fp}")
        song_tab.save_to_file(tab_fp)

        # Convert to note sequence
        mt = song_tab.convert_to_note_seq()
        # more_accurate_note_seq = duration_renderer(out_note_seq)

        # Set tempo
        mt.set_tempo(90)

        # Save the note sequence to MIDI
        midi_fp = jpath(save_dir, f"{save_name}.mid")
        print(f"Saving MIDI to {midi_fp}")
        mt.to_midi(midi_fp)

        # Synthesize the MIDI to WAV
        sf_path = "resources/Tyros Nylon.sf2"
        audio_fp = jpath(save_dir, f"{save_name}.wav")
        print(f"Synthesizing MIDI to WAV: {audio_fp}")
        self.midi_to_wav(midi_fp, sf_path, audio_fp)
        self.post_process_wav(
            audio_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1
        )

    def normalize_melody_range(self, melody_of_bars: List[NoteSeq]):
        """
        Normalize the melody range to a specific range
        Target range: D3 (MIDI 50) to G5 (MIDI 79)
        """
        LOW_NOTE = "A2"  # D3 (4th string) or A2 (5th string)
        HIGH_NOTE = "A5"
        LOW = note_name_to_midi_pitch(LOW_NOTE)
        HIGH = note_name_to_midi_pitch(HIGH_NOTE)
        print(f"Normalizing melody range to {LOW_NOTE} - {HIGH_NOTE} ({LOW} - {HIGH})")
        for note_seq in melody_of_bars:
            for note in note_seq.notes:
                while note.pitch < LOW:
                    note.pitch += 12
                    print("shift up")
                while note.pitch > HIGH:
                    note.pitch -= 12
                    print("shift down")
        return melody_of_bars

    def midi_to_note_seq(self, midi_fp):
        """
        Convert MIDI file to note sequence
        """
        mt = MultiTrack.from_midi(midi_fp)
        mt.quantize_to_16th()
        note_seq_per_bar = mt.get_all_notes_by_bar()
        mt.get_note_list

        melody = mt.get_melody("hi_track")

        return note_seq_per_bar

    def extract_melody(self, method="hi_track_song") -> List[NoteSeq]:
        """
        Extract melody from the multi-track MIDI.

        method options:
          'hi_track_song' - track with highest avg pitch across the whole song
          'hi_track'      - track with highest avg pitch, decided per bar
          'hi_note'       - highest-pitched note at each onset, per bar
          'hi_note_dur'   - like hi_note, but drops notes that overlap a previous note
          'inst_id_XX'    - all notes from instrument ID XX (e.g. 'inst_id_25')
        """
        if method == "hi_track_song":
            melody = self.mt.get_melody_of_song("hi_track")
        elif method in ("hi_track", "hi_note", "hi_note_dur"):
            melody = self.mt.get_melody(method)
        elif method.startswith("inst_id_"):
            inst_id = int(method.split("_")[-1])
            melody = self.mt.get_all_notes_by_bar(of_insts=[inst_id])
        else:
            raise ValueError(f"Unknown melody_method: {method!r}")

        return [NoteSeq(notes) for notes in melody]

    def get_all_notes(self):
        notes_of_bars = self.mt.get_all_notes_by_bar(include_drum=False)
        ret = []
        for notes_of_bar in notes_of_bars:
            ret.append(NoteSeq(notes_of_bar))
        return ret

    def extract_chord(self):
        chords = [bar.get_chord() for bar in self.mt]
        ret = []
        for chord_of_bar in chords:
            ret.append(ChordSeq(chord_of_bar))
        return ret

    def midi_to_wav(self, midi_fp, sf_path, wav_fp):
        cmd = ["fluidsynth", "-ni", sf_path, midi_fp, "-F", wav_fp, "-r", "44100"]
        subprocess.run(cmd, check=True)

    def post_process_wav(
        self, wav_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1
    ):
        audio = AudioSegment.from_wav(wav_fp)

        # Step 1: Normalize to target dBFS
        change_in_dBFS = target_dbfs - audio.max_dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        # Step 2: Trim silence after normalization
        start_trim = self.detect_leading_silence(normalized, silence_thresh_db)
        end_trim = self.detect_leading_silence(normalized.reverse(), silence_thresh_db)
        duration = len(normalized)
        trimmed = normalized[start_trim : duration - end_trim + padding_ms]

        # Export
        trimmed.export(wav_fp, format="wav")

    def detect_leading_silence(self, sound, silence_threshold=-40.0, chunk_size=10):
        trim_ms = 0
        while (
            trim_ms < len(sound)
            and sound[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold
        ):
            trim_ms += chunk_size
        return trim_ms


if __name__ == "__main__":
    main()
