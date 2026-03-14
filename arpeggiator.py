from remi_z import NoteSeq
from typing import List
from fretboard_util import Fretboard
from tab_util import Chart, Tab, TabSeq
import random


class Arpeggiator:
    '''
    Takes in a sequence of chart, and a note sequence as texture reference,
    Generate tab
    '''
    def __init__(self, time_res=16):
        self.fretboard = Fretboard()
        self.time_res = time_res # default to 16 positions per bar, 16th note resolution

    def calculate_groove_for_a_bar(note_seq):
        '''
        groove is represented by
        onset position of bass note
        onset position of melody note
        counter melody onset position (highest note of filling)
        filling note onset density of each position (except melody and bass onset)
        '''
        pass

    def arpeggiate_a_bar(self, melody: NoteSeq, chart_list_of_the_bar: List[Chart], notes_of_the_bar: NoteSeq) -> Tab:
        '''
        Generate a guitar tab for one bar by layering three voices:
          1. Melody  – placed on the string matching each melody note in the chart.
          2. Bass    – chord root on lowest string, placed at bass note onsets.
          3. Fillings – inner voices at remaining active positions, assigned to
                        higher or lower inner strings based on pitch contour.
        '''
        assert len(chart_list_of_the_bar) == 2, "Expected exactly 2 charts per bar."
        assert len(notes_of_the_bar.notes) > 0, "Bar has no notes."

        MAX_POSITIONS = self.time_res
        HALF = MAX_POSITIONS // 2
        coeff = 48 // MAX_POSITIONS

        chart_1, chart_2 = chart_list_of_the_bar
        notes_first_half  = NoteSeq([n for n in notes_of_the_bar.notes if n.onset < 24])
        notes_second_half = NoteSeq([n for n in notes_of_the_bar.notes if n.onset >= 24])
        melody_first_half  = NoteSeq([n for n in melody.notes if n.onset < 24])
        melody_second_half = NoteSeq([n for n in melody.notes if n.onset >= 24])

        tab = Tab(n_positions=MAX_POSITIONS)

        # --- Step 1: Melody layer ---
        melody_psf = (self.get_psf_from_chart(melody_first_half, chart_1) +
                      self.get_psf_from_chart(melody_second_half, chart_2))
        for pos, string_id, fret, _ in melody_psf:
            tab.add_note(pos, string_id, fret)

        # --- Step 2: Bass layer ---
        bass_seq_1 = get_bass_note_seq(notes_first_half)
        bass_seq_2 = get_bass_note_seq(notes_second_half)
        bass_onset_pos_1 = [n.onset // coeff for n in bass_seq_1.notes]
        bass_onset_pos_2 = [n.onset // coeff for n in bass_seq_2.notes]
        bass_sf_1 = self.get_bass_sf_from_chart(chart_1)
        bass_sf_2 = self.get_bass_sf_from_chart(chart_2)

        bass_psf = []
        for pos in bass_onset_pos_1:
            string_id, fret = bass_sf_1
            tab.add_note(pos, string_id, fret)
            bass_psf.append((pos, string_id, fret))
        for pos in bass_onset_pos_2:
            string_id, fret = bass_sf_2
            tab.add_note(pos, string_id, fret)
            bass_psf.append((pos, string_id, fret))

        tab.add_chord(0, chart_1.chord_name)
        tab.add_chord(HALF, chart_2.chord_name)

        # --- Step 3: Filling layer ---
        bass_notes = NoteSeq(bass_seq_1.notes + bass_seq_2.notes)
        fillings = self.get_filling_notes(notes_of_the_bar, melody, bass_notes)

        melody_positions = set(n.onset // coeff for n in melody.notes)
        note_positions   = set(n.onset // coeff for n in notes_of_the_bar.notes)
        filling_positions = sorted(note_positions - melody_positions)

        if not filling_positions or not fillings.notes:
            return tab

        # Build per-position string maps (propagated forward to cover silent positions)
        melody_strings = self._build_propagated_string_map(
            [(pos, sid, fret) for pos, sid, fret, _ in melody_psf]
        )
        bass_strings = self._build_propagated_string_map(bass_psf)

        # Assign each filling position to a high (1) or low (0) inner voice
        sub_melody_contour = self.get_filling_submelody_contour(fillings)
        voice_assign = self._assign_filling_voices(filling_positions, sub_melody_contour)

        # Place filling notes on the chosen string from the active chart,
        # avoiding repeating the same (string, fret) as the immediately preceding position.
        for pos in filling_positions:
            chart = chart_1 if pos < HALF else chart_2
            string_id = self._pick_filling_string(pos, voice_assign, melody_strings, bass_strings)
            candidates = [sf for sf in chart.string_fret_list if sf[0] == string_id]
            if not candidates:
                continue
            # Exclude any candidate already placed at the previous position on that string.
            # If no alternative exists, skip this position entirely rather than repeating.
            if pos > 0:
                prev_fret = tab.matrix[string_id - 1, pos - 1]
                fresh = [sf for sf in candidates if sf[1] != prev_fret]
                if not fresh:
                    continue
                candidates = fresh
            sf = random.choice(candidates)
            tab.add_note(pos, sf[0], sf[1])

        return tab

    def _build_propagated_string_map(self, psf_list: list) -> list:
        '''
        Build a per-position string-ID list from (pos, string_id, fret) tuples.
        Positions with no entry forward-fill from the previous known string.
        '''
        string_map = [-1] * self.time_res
        for pos, string_id, _ in psf_list:
            if string_id > 0:
                string_map[pos] = string_id
        for pos in range(1, self.time_res):
            if string_map[pos] == -1:
                string_map[pos] = string_map[pos - 1]
        return string_map

    def _assign_filling_voices(self, filling_positions: list, sub_melody_contour: dict) -> list:
        '''
        Classify each filling position as high voice (1) or low voice (0) by comparing
        the contour pitch at that position against the mean contour pitch.
        '''
        sub_mel_pitch = [0] * self.time_res
        for pos in filling_positions:
            note = sub_melody_contour.get(pos)
            if note is not None:
                sub_mel_pitch[pos] = note.pitch

        pitches = [p for p in sub_mel_pitch if p > 0]

        voice_assign = [-1] * self.time_res
        if not pitches:
            # No contour data — assign everything to low voice
            for pos in filling_positions:
                voice_assign[pos] = 0
            return voice_assign

        mean_pitch = sum(pitches) / len(pitches)
        for pos in filling_positions:
            voice_assign[pos] = 1 if sub_mel_pitch[pos] > mean_pitch else 0
        return voice_assign

    def _pick_filling_string(self, pos: int, voice_assign: list,
                             melody_strings: list, bass_strings: list) -> int:
        '''
        Choose a string for the filling note at `pos`.

        String IDs run 6 (low E) → 1 (high e). Usable strings are those strictly
        between the bass string and the melody string. The voice assignment (high=1,
        low=0) determines which end of the usable range to prefer.

        Fallback rules when usable strings are scarce:
          - 1 usable string: high voice → melody string, low voice → one above bass.
          - 0 usable strings: high voice → melody string, low voice → bass string.
        '''
        voice      = voice_assign[pos]
        mel_string  = melody_strings[pos]
        bass_string = bass_strings[pos]

        usable = set(range(1, 7))
        if bass_string > 0:
            usable -= set(range(bass_string, 7))    # exclude bass and lower strings
        if mel_string > 0:
            usable -= set(range(1, mel_string + 1)) # exclude melody and higher strings

        if len(usable) >= 2:
            return bass_string - 2 if voice == 1 else bass_string - 1
        elif len(usable) == 1:
            return mel_string if voice == 1 else bass_string - 1
        else:
            return mel_string if voice == 1 else bass_string

    def get_psf_from_chart(self, melody: 'NoteSeq', chart: 'Chart') -> list:
        """
        Find every position-string-fret pair for each melody note in the chart.

        Args:
            melody: NoteSeq object containing melody notes (each note must have .get_note_name()).
            chart: Chart object containing string-fret mapping for the bar.

        Returns:
            List of tuples (string_id, fret, note_name) for each melody note found in the chart.
        """
        coeff = 48 // self.time_res
        result = []

        for melody_note in melody:
            pos = melody_note.onset // coeff  # Convert to 8th note position
            note_name = melody_note.get_note_name()

            # Find the string-fret pair for this melody note in the chart
            sf = chart.get_sf_from_note_name(note_name)
            assert sf is not None, f"Melody note {note_name} not found in chart."
            string_id, fret = sf
            result.append((pos, string_id, fret, note_name))
        return result


    def get_bass_sf_from_chart(self, chart: 'Chart') -> list:
        """
        Get the string-fret pair for the bass note (chord root) from the chart.

        Args:
            chart: Chart object containing string-fret mapping for the bar.

        Returns:
            List of tuples (string_id, fret) for the bass note.
        """
        # Return the SF with lowest string (largest string ID), and then lowest fret (smallest fret number)
        all_sfs = chart.string_fret_list
        bass_sf = min(all_sfs, key=lambda sf: (-sf[0], sf[1]))
        return bass_sf


    def arpeggiate_a_song(self, melody, chart_list_of_the_song, note_seq_of_the_song) -> TabSeq:
        # Group chart_list by bar
        chart_list_of_bars = []
        chart_list_of_bar = []
        for chart in chart_list_of_the_song:
            if len(chart_list_of_bar) < 2:
                chart_list_of_bar.append(chart)
                if len(chart_list_of_bar) == 2:
                    chart_list_of_bars.append(chart_list_of_bar)
                    chart_list_of_bar = []

        # Call arpeggiate_a_bar function to do the job for each bar.
        tab_of_the_bars = []
        for melody_of_bar, chart_list_of_bar, note_seq_of_bar in zip(melody, chart_list_of_bars, note_seq_of_the_song):
            tab_of_the_bar = self.arpeggiate_a_bar(melody_of_bar, chart_list_of_bar, note_seq_of_bar)
            tab_of_the_bars.append(tab_of_the_bar)
        tab_of_the_song = TabSeq(tab_of_the_bars, tab_per_row=2)

        return tab_of_the_song

    def get_filling_submelody_contour(self, filling_notes: 'NoteSeq') -> dict:
        """
        Calculate the filling sub-melody contour: for each onset position,
        find the highest note (by pitch) among filling notes.

        Args:
            filling_notes: NoteSeq object containing only filling notes (each note must have .onset and .pitch).

        Returns:
            Dictionary mapping onset position (int) to the highest filling note (Note object).
        """
        coeff = 48 // self.time_res

        contour = {}
        onset_dict = {}
        for note in filling_notes.notes:
            onset = note.onset // coeff  # Convert to 8th note position if needed
            onset_dict.setdefault(onset, []).append(note)
        for onset, notes_at_onset in onset_dict.items():
            highest_note = max(notes_at_onset, key=lambda n: n.pitch)
            contour[onset] = highest_note
        return contour

    def get_filling_note_density_by_position(self, filling_notes: 'NoteSeq') -> dict:
        """
        Calculate the filling note density for each onset position directly from filling_notes.

        Args:
            filling_notes: NoteSeq object containing only filling notes (each note must have .onset).

        Returns:
            Dictionary mapping onset position (int) to filling note density (int).
        """
        coeff = 48 // self.time_res

        density_by_position = {}
        for note in filling_notes.notes:
            onset = note.onset // coeff  # Convert to 8th note position if needed
            density_by_position[onset] = density_by_position.get(onset, 0) + 1
        return density_by_position

    def get_filling_notes(self, note_seq: 'NoteSeq', melody_seq: 'NoteSeq', bass_seq: 'NoteSeq') -> 'NoteSeq':
        """
        Return a NoteSeq containing notes that are neither melody nor bass notes.

        Args:
            note_seq: NoteSeq object containing all notes.
            melody_seq: NoteSeq object containing melody notes.
            bass_seq: NoteSeq object containing bass notes.

        Returns:
            NoteSeq containing only filling notes.
        """
        melody_set = set((note.onset, note.pitch) for note in melody_seq.notes)
        bass_set = set((note.onset, note.pitch) for note in bass_seq.notes)
        filling_notes = [
            note for note in note_seq.notes
            if (note.onset, note.pitch) not in melody_set and (note.onset, note.pitch) not in bass_set
        ]
        return NoteSeq(filling_notes)


class DurationRenderer:
    '''
    Modify the duration of the system to make them sounds more natural and practical
    '''
    def modify_duration(note_seq):
        return note_seq


def get_bass_note_seq(note_seq: 'NoteSeq', chord_root: str = None) -> 'NoteSeq':
    """
    Given a NoteSeq, find the note(s) with the lowest pitch among all notes.
    Return a new NoteSeq containing all notes with that lowest pitch.

    Args:
        note_seq: NoteSeq object containing notes (each note must have .onset, .get_note_name(), .pitch).
        chord_root: (Unused) The root note name of the chord (e.g., "C", "G#", "A").

    Returns:
        NoteSeq containing all notes with the lowest pitch.
    """
    if not note_seq.notes:
        return NoteSeq([])
    min_pitch = min(note.pitch for note in note_seq.notes)
    bass_notes = [note for note in note_seq.notes if note.pitch == min_pitch]
    return NoteSeq(bass_notes)


def get_bass_note_seq_old(note_seq: 'NoteSeq', chord_root: str) -> 'NoteSeq':
    """
    Old version, bass's definition consider both lowest note and the chord root

    Given a NoteSeq, find the bass note (lowest pitch) at each onset position
    whose pitch class matches the chord root, and return a new NoteSeq containing
    these bass notes. Only keep bass notes in the most common octave among matched notes.

    Args:
        note_seq: NoteSeq object containing notes (each note must have .onset, .get_note_name(), .pitch).
        chord_root: The root note name of the chord (e.g., "C", "G#", "A").

    Returns:
        NoteSeq containing the bass notes for each onset position, filtered by octave.
    """
    bass_notes = []
    onset_dict = {}
    for note in note_seq.notes:
        onset_dict.setdefault(note.onset, []).append(note)
    for onset, notes_at_onset in onset_dict.items():
        matching_notes = [note for note in notes_at_onset if chord_root in note.get_note_name()]
        if matching_notes:
            bass_note = min(matching_notes, key=lambda n: n.pitch)
            bass_notes.append(bass_note)
    # Filter by most common octave
    if bass_notes:
        # Extract octave from note name (assume last char is octave, e.g. "C3")
        octaves = [int(note.get_note_name()[-1]) for note in bass_notes]
        from collections import Counter
        most_common_octave = Counter(octaves).most_common(1)[0][0]
        bass_notes = [note for note in bass_notes if int(note.get_note_name()[-1]) == most_common_octave]
    return NoteSeq(bass_notes)
