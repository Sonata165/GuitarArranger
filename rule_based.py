'''
这个文件用来描述Rule-Based Guitar Arranger的伪代码
'''
from remi_z import MultiTrack, NoteSeq, ChordSeq, midi_pitch_to_note_name, Note, note_name_to_midi_pitch
from typing import List
from fretboard_util import Fretboard
from tab_util import Chart, Tab, TabSeq
import fluidsynth
import subprocess
from pydub import AudioSegment
from sonata_utils import jpath, create_dir_if_not_exist
import random


def main():
    arranger = ArrangerSystem()
    midi_fp = 'misc/midi_normalized/caihong.mid'
    # midi_fp = 'misc/canon_in_D.mid'
    arranger.arrange_song_from_midi(midi_fp)


class ArrangerSystem:
    def __init__(self):
        self.mt:MultiTrack = None
        self.voicer = Voicer()
        self.arpeggiator = Arpeggiator()

    def arrange_song_from_midi(self, midi_fp):
        song_name = midi_fp.split('/')[-1].split('.')[0]
        print(f'Arranging song: {song_name}')
        
        bar_start = 4
        bar_duration = 8
        bar_end = bar_start + bar_duration

        save_name = f'{song_name}_bar_{bar_start}_{bar_end}'
        save_dir = jpath('outputs', song_name, f'{save_name}')
        create_dir_if_not_exist(save_dir)
        
        # self.mt = MultiTrack.from_midi(midi_fp)[4:12] # [0:1]
        # self.mt = MultiTrack.from_midi(midi_fp)[12:20] # [0:1]
        # self.mt = MultiTrack.from_midi(midi_fp)[20:28] # [0:1]
        # self.mt = MultiTrack.from_midi(midi_fp)[28:36] # [0:1]
        # self.mt = MultiTrack.from_midi(midi_fp)[44:52] # [0:1]

        self.mt = MultiTrack.from_midi(midi_fp)[bar_start:bar_end]

        self.mt.quantize_to_16th()
        # self.mt.shift_pitch(-5)
        # self.mt.shift_pitch(-12)
        # self.mt.shift_pitch(-7)
        # self.mt.shift_pitch(2)

        # Prepare model inputs
        notes_of_bars = self.get_all_notes()
        melody = self.extract_melody() # a list of NoteSeq
        chord_of_bars = self.extract_chord()

        # Melody range normalization
        melody = self.normalize_melody_range(melody)
        # Save melody to file
        melody_fp = jpath(save_dir, f'{save_name}_melody.mid')
        melody_mt = MultiTrack.from_note_seqs(melody, program_id=25) # 24在garadgeband里全是滑音，太奇怪了
        print(f'Saving melody to {melody_fp}')
        melody_mt.to_midi(melody_fp)

        # Synthesize the melody to WAV
        sf_path = 'resources/Tyros Nylon.sf2'
        melody_audio_fp = jpath(save_dir, f'{save_name}_melody.wav')
        print(f'Synthesizing melody MIDI to WAV: {melody_audio_fp}')
        self.midi_to_wav(melody_fp, sf_path, melody_audio_fp)
        self.post_process_wav(melody_audio_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1)

        # Get chart sequence (left hand modeling)
        chart_seq = self.voicer.generate_chart_sequence_for_song(melody, chord_of_bars)
        if not chart_seq or len(chart_seq) == 0:
            raise ValueError('No chart sequence generated. Please check model robustness.')
        chart_fp = jpath(save_dir, f'{save_name}_chart.txt')
        print(f'Saving chart sequence to {chart_fp}')
        self.voicer.save_chart_sequence_to_file(chart_seq, chart_fp)

        # Tab generation given charts (right hand modeling)
        song_tab = self.arpeggiator.arpeggiate_a_song(melody, chart_seq, notes_of_bars)

        # Save the tab to file
        tab_fp = jpath(save_dir, f'{save_name}_tab.txt')
        print(f'Saving tab to {tab_fp}')
        song_tab.save_to_file(tab_fp)

        # Convert to note sequence
        mt = song_tab.convert_to_note_seq()
        # more_accurate_note_seq = duration_renderer(out_note_seq)

        # Set tempo
        mt.set_tempo(90)

        # Save the note sequence to MIDI
        # midi_fp = 'test_out.mid'
        midi_fp = jpath(save_dir, f'{save_name}.midi')
        print(f'Saving MIDI to {midi_fp}')
        mt.to_midi(midi_fp)

        # Synthesize the MIDI to WAV
        sf_path = 'resources/Tyros Nylon.sf2'
        audio_fp = jpath(save_dir, f'{save_name}.wav')
        # wav_fp = 'test_out.wav'
        print(f'Synthesizing MIDI to WAV: {audio_fp}')
        self.midi_to_wav(midi_fp, sf_path, audio_fp)
        self.post_process_wav(audio_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1)


    def normalize_melody_range(self, melody_of_bars:List[NoteSeq]):
        '''
        Normalize the melody range to a specific range
        Target range: D3 (MIDI 50) to G5 (MIDI 79)
        '''
        LOW_NOTE = 'A2' # D3 (4th string) or A2 (5th string)
        HIGH_NOTE = 'A5'
        LOW = note_name_to_midi_pitch(LOW_NOTE)
        HIGH = note_name_to_midi_pitch(HIGH_NOTE)
        print(f'Normalizing melody range to {LOW_NOTE} - {HIGH_NOTE} ({LOW} - {HIGH})')
        for note_seq in melody_of_bars:
            for note in note_seq.notes:
                while note.pitch < LOW:
                    note.pitch += 12
                    print('shift up')
                while note.pitch > HIGH:
                    note.pitch -= 12
                    print('shift down')
        return melody_of_bars

    
    def midi_to_note_seq(self, midi_fp):
        '''
        Convert MIDI file to note sequence
        '''
        mt = MultiTrack.from_midi(midi_fp)
        mt.quantize_to_16th()
        note_seq_per_bar = mt.get_all_notes_by_bar()
        mt.get_note_list

        melody = mt.get_melody('hi_track')

        return note_seq_per_bar
    
    def extract_melody(self) -> List[NoteSeq]:
        '''
        Naively extract melody from the multi-track MIDI
        
        We assume the track with the highest average pitch is the melody track.
        '''
        melody = self.mt.get_melody_of_song('hi_track')   # Highest track in each bar
        # melody = self.mt.get_melody('hi_track')   # Highest track in each bar
        # melody = self.mt.get_melody('hi_note')  # Highest note in each position
        ret = []
        for melody_of_bar in melody:
            ret.append(NoteSeq(melody_of_bar))
        return ret

    def get_all_notes(self):
        notes_of_bars = self.mt.get_all_notes_by_bar()
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
        cmd = [
            "fluidsynth",
            "-ni",
            sf_path,
            midi_fp,
            "-F", wav_fp,
            "-r", "44100"
        ]
        subprocess.run(cmd, check=True)

    def post_process_wav(self, wav_fp, silence_thresh_db=-40.0, padding_ms=200, target_dbfs=-0.1):
        audio = AudioSegment.from_wav(wav_fp)

        # Step 1: Normalize to target dBFS
        change_in_dBFS = target_dbfs - audio.max_dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        # Step 2: Trim silence after normalization
        start_trim = self.detect_leading_silence(normalized, silence_thresh_db)
        end_trim = self.detect_leading_silence(normalized.reverse(), silence_thresh_db)
        duration = len(normalized)
        trimmed = normalized[start_trim:duration - end_trim + padding_ms]

        # Export
        trimmed.export(wav_fp, format="wav")


    def detect_leading_silence(self, sound, silence_threshold=-40.0, chunk_size=10):
        trim_ms = 0
        while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
            trim_ms += chunk_size
        return trim_ms

class Block:
    def __init__(self, chord:tuple, melody:NoteSeq):
        self.chord = chord
        self.melody = melody

    def __str__(self):
        return f'Chord: {self.chord}, Melody: {self.melody}'
    
    def __repr__(self):
        return self.__str__()


class Voicer:
    def __init__(self):
        self.fretboard = Fretboard()

    def generate_chart_candidate_for_block(self, block:Block):
        '''
        生成chart候选 for a block
        Block means all melody covered by a same chord
        or half of a bar
        which is shorter.

        Input:
        - melody_notes: list of notes in the melody, e.g. ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
        - chord: a chord string, e.g. "Cmaj7"

        Return:
        - chart_candidate: a list of chart candidates, e.g., [chart1, chart2, chart3]
        Each chart indicate a position on guitar neck and all string-fret that contains melody notes and all chord notes.

        '''
        # Find all possible positions to fret these melody notes
        melody = block.melody.get_note_name_list()
        possible_positions = self.fretboard.find_all_playable_position_of_note_set(melody)
        melody_pitch_range = block.melody.get_pitch_range()
        if melody_pitch_range is None:
            chord_upper_pitch_limit = note_name_to_midi_pitch('D5')
        else:
            chord_upper_pitch_limit = melody_pitch_range[0]
        lowest_melody_note = midi_pitch_to_note_name(chord_upper_pitch_limit)

        # If no possible position, raise warning
        # print(len(possible_positions))
        if len(possible_positions) == 0:
            print(f'Warning: No possible position found for melody {melody} with chord {block.chord}.')
            return []

        # For each position
        ret = []
        for position in possible_positions:
            # Find all chord note in that position (try press)
            chord_str = f'{block.chord[0]}{block.chord[1]}'
            chord_note_sfs = self.fretboard.press_chord(chord_str, 
                                                        position=position, 
                                                        string_press_once=False, 
                                                        enforce_root=True,
                                                        closed=False,
                                                        force_highest_pitch=lowest_melody_note,
                                                        )
            
            if chord_str == 'D': # debug
                a = 2

            # Press the melody notes in that position
            melody_note_sfs = self.fretboard.press_note_set(note_set=melody, lowest_fret=position)

            # If fail to press this chord, skip this position
            # We want at least one string that is not used by melody
            melody_strings = set([sf[0] for sf in melody_note_sfs])
            thres = 1 # at least we want {thres} useable strings in the chord note
            chord_strings = set([sf[0] for sf in chord_note_sfs])
            # get non-melody strings in chord
            non_mel_chord_strings = chord_strings - melody_strings
            if len(non_mel_chord_strings) < thres:
                continue

            # Draw a chart, add chord notes
            chord_name = f'{block.chord[0]}{block.chord[1]}'
            chart = Chart(string_fret_list=chord_note_sfs, chord_name=chord_name)

            # Add the melody notes to the chart
            melody_str = ' '.join(melody)
            chart.fret_more_note(melody_note_sfs, melody_list=melody_str)
            ret.append(chart)

        # If no chart candidate, raise warning
        if len(ret) == 0:
            print(f'Warning: No chart candidate found for melody {block.melody} with chord {block.chord}.')
            print('possible positions for melody only:', possible_positions)
            return []

        return ret

    def generate_chart_candidate_sequence_for_bar(self, melody_of_bar:NoteSeq, chord_of_bar:ChordSeq):
        '''
        Return a chart candidate sequence
        '''
        # Break melody and chord into blocks
        # Here we assume half a bar is a block
        melody_of_block_1 = NoteSeq([note for note in melody_of_bar.notes if note.onset<24])
        chord_of_block_1 = chord_of_bar.chord_list[0]
        melody_of_block_2 = NoteSeq([note for note in melody_of_bar.notes if note.onset>=24])
        chord_of_block_2 = chord_of_bar.chord_list[1]
        block_1 = Block(chord = chord_of_block_1, melody=melody_of_block_1)
        block_2 = Block(chord=chord_of_block_2, melody=melody_of_block_2)
        blocks = [block_1, block_2]

        ret = []
        for block in blocks:
            chart_candidates_of_block = self.generate_chart_candidate_for_block(block)
            ret.append(chart_candidates_of_block)    

        return ret

    def find_best_chart_path(self, chart_candidates_seq):
        '''
        Find the best path for the song
        That minimize the left hand position movement
        Uses Dijkstra's algorithm to find the path with minimal sum of avg_fret differences.
        '''
        import heapq
        if not chart_candidates_seq or not all(chart_candidates_seq):
            return []
        n_layers = len(chart_candidates_seq)
        # Each node is (layer_idx, chart_idx)
        # Dijkstra: (total_cost, layer_idx, chart_idx, path_so_far)
        heap = []
        for idx, chart in enumerate(chart_candidates_seq[0]):
            heapq.heappush(heap, (0, 0, idx, [idx]))  # cost, layer, chart_idx, path (as indices)
        # best_cost[layer][chart_idx] = cost
        best_cost = [{} for _ in range(n_layers)]
        for idx in range(len(chart_candidates_seq[0])):
            best_cost[0][idx] = 0
        while heap:
            cost, layer, idx, path = heapq.heappop(heap)
            if layer == n_layers - 1:
                # Reached last layer, connect to end node (cost 0)
                return [chart_candidates_seq[i][j] for i, j in enumerate(path)]
            current_chart = chart_candidates_seq[layer][idx]
            for next_idx, next_chart in enumerate(chart_candidates_seq[layer + 1]):
                edge_cost = abs(current_chart.avg_fret - next_chart.avg_fret)
                new_cost = cost + edge_cost
                if next_idx not in best_cost[layer + 1] or new_cost < best_cost[layer + 1][next_idx]:
                    best_cost[layer + 1][next_idx] = new_cost
                    heapq.heappush(heap, (new_cost, layer + 1, next_idx, path + [next_idx]))
        # If we get here, no path found
        return []

    def generate_chart_sequence_for_song(self, melody_of_bars:List[NoteSeq], chord_progression):
        '''
        Find the best chart sequence for the melody and chord progression of a song.
        That minimize the left-hand position-wise movement on neck

        Implemented by a shortest path algorithm.
        '''
        assert len(melody_of_bars) == len(chord_progression)
        chart_candidates = []
        for melody_of_bar, chord_of_bar in zip(melody_of_bars, chord_progression):
            chart_candidates.extend(self.generate_chart_candidate_sequence_for_bar(melody_of_bar, chord_of_bar))
        
        best_chart_seq = self.find_best_chart_path(chart_candidates)

        return best_chart_seq

    def save_chart_sequence_to_file(self, chart_sequence, filename):
        '''
        Save the chart sequence to a .chart file as plain text, using __str__ of each Chart,
        and including chord/melody info if present.
        '''
        with open(filename, 'w', encoding='utf-8') as f:
            for i, chart in enumerate(chart_sequence):
                f.write(f'Chart {i+1}\n')
                f.write(f'Chord: {chart.chord_name}\n')
                f.write(f'Melody: {chart.melody_list}\n')
                f.write(str(chart))
                f.write('\n')
                f.write('-' * 40 + '\n')


class Arpeggiator:
    '''
    Takes in a sequence of chart, and a note sequence as texture reference,
    Generate tab
    '''
    def __init__(self):
        self.fretboard = Fretboard()
        
    def calculate_groove_for_a_bar(note_seq):
        '''
        groove is represented by 
        onset position of bass note
        onset position of melody note
        counter melody onset position (highest note of filling)
        filling note onset density of each position (except melody and bass onset)
        '''
        pass

    def arpeggiate_a_bar(self, melody, chart_list_of_the_bar:List[Chart], notes_of_the_bar:NoteSeq):
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。

        '''
        notes_first_half = NoteSeq([note for note in notes_of_the_bar.notes if note.onset < 24])
        notes_second_half = NoteSeq([note for note in notes_of_the_bar.notes if note.onset >= 24])
        melody_first_half = NoteSeq([note for note in melody.notes if note.onset < 24])
        melody_second_half = NoteSeq([note for note in melody.notes if note.onset >= 24])
        assert len(chart_list_of_the_bar) == 2, "There should be exactly 2 charts for a bar."
        assert len(notes_of_the_bar.notes) > 0, "There should be notes in the bar."

        # Feature extraction, getting 

        # 1. Melody note onset position
        melody_onset_positions = [note.onset // 6 for note in melody.notes]

        # 2. Bass note onset position
        bass_note_1_class = chart_list_of_the_bar[0].chord_name[:1] if '#' not in chart_list_of_the_bar[0].chord_name else chart_list_of_the_bar[0].chord_name[:2]
        bass_note_2_class = chart_list_of_the_bar[1].chord_name[:1] if '#' not in chart_list_of_the_bar[1].chord_name else chart_list_of_the_bar[1].chord_name[:2]
        bass_note_1 = get_bass_note_seq(notes_first_half, bass_note_1_class)
        bass_note_2 = get_bass_note_seq(notes_second_half, bass_note_2_class)
        bass_notes = NoteSeq(bass_note_1.notes + bass_note_2.notes)
        bass_onset_positions = [note.onset // 6 for note in bass_notes.notes]
        bass_note_onset_pos_1 = [note.onset // 6 for note in bass_note_1.notes]
        bass_note_onset_pos_2 = [note.onset // 6 for note in bass_note_2.notes]

        # 3. Filling note density by position
        fillings = get_filling_notes(notes_of_the_bar, melody, bass_notes)
        fillings_density_by_position = get_filling_note_density_by_position(fillings)

        # 4. Filling sub-melody contour (highest note of filling notes in each onset position)
        sub_melody_contour = get_filling_submelody_contour(fillings)

        # Generate an empty tab
        tab = Tab()

        # Fill melody note to melody note position
        chart_1 = chart_list_of_the_bar[0]
        chart_2 = chart_list_of_the_bar[1]
        psf_1 = self.get_psf_from_chart(melody_first_half, chart_1)
        psf_2 = self.get_psf_from_chart(melody_second_half, chart_2)
        melody_psf = psf_1 + psf_2
        # Add melody notes to the tab
        for psf in melody_psf:
            pos, string_id, fret, note_name = psf
            tab.add_note(pos, string_id, fret)
        # for psf in psf_1:
        #     pos, string_id, fret, note_name = psf
        #     tab.add_note(pos, string_id, fret)
        # for psf in psf_2:
        #     pos, string_id, fret, note_name = psf
        #     tab.add_note(pos, string_id, fret)
        
        
        # Fill bass note to bass position
        # Get string-fret pair for bass notes (chord root) from the chart
        bass_notes_psf = []
        bass_sf_1 = self.get_bass_sf_from_chart(chart_1)
        bass_sf_2 = self.get_bass_sf_from_chart(chart_2)
        for pos in bass_note_onset_pos_1:
            string_id, fret = bass_sf_1
            tab.add_note(pos, string_id, fret)
            bass_notes_psf.append((pos, string_id, fret))
        for pos in bass_note_onset_pos_2:
            string_id, fret = bass_sf_2
            tab.add_note(pos, string_id, fret)
            bass_notes_psf.append((pos, string_id, fret))

        # Add chord name to the tab
        chord_1 = chart_1.chord_name
        chord_2 = chart_2.chord_name
        tab.add_chord(0, chord_1)
        tab.add_chord(4, chord_2)

        # Add fills: 
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，在没有弹奏任何主旋律和低音的情况下，放上剩下的音符中最高音的音符

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。
        '''
        # Step 1: Find all positions that need filling notes (positions that does not have melody notes)
        all_positions = set(range(0, 8))
        melody_positions = set(melody_onset_positions)
        filling_positions = all_positions - melody_positions # - set(bass_onset_positions)
        filling_positions = list(filling_positions)
        filling_positions.sort()  # Sort positions for consistent processing

        # If no filling positions, return the tab with melody and bass notes only
        if len(filling_positions) == 0:
            return tab

        # Check melody notes occupy which string in each position
        melody_strings_used_by_position = [-1]*8 # String 6~1 means lowest to highest. 0 means no melody note
        for psf in melody_psf:
            pos, string_id, fret, note_name = psf
            if string_id > 0:
                melody_strings_used_by_position[pos] = string_id
        # Fill in melody strings for positions without melody notes
        for pos in range(1, len(melody_strings_used_by_position)):
            prev_melody_string = melody_strings_used_by_position[pos - 1]
            if melody_strings_used_by_position[pos] == -1:
                # If no melody string used, use the previous melody string
                melody_strings_used_by_position[pos] = prev_melody_string
            
        # Check which string is occupied by bass note for each filling position
        bass_strings_used_by_position = [-1]*8 # String 6~1 means lowest to highest. 0 means no bass note

        # Check if any usable filling notes in the chart (higher than bass string)
        for psf in bass_notes_psf:
            pos, string_id, fret = psf
            if string_id > 0:  
                bass_strings_used_by_position[pos] = string_id
        for pos in range(1, len(bass_strings_used_by_position)):
            prev_bass_string = bass_strings_used_by_position[pos - 1]
            if bass_strings_used_by_position[pos] == -1:
                # If no bass string used, use the previous bass string
                bass_strings_used_by_position[pos] = prev_bass_string

        # Step 2: Determine the countour of filling notes
        sub_mel_pitch = [0] * 8
        for pos in filling_positions:
            # Get the highest note in the filling contour at this position
            highest_note = sub_melody_contour.get(pos, None)
            if highest_note is not None:
                sub_mel_pitch[pos] = highest_note.pitch
        all_sub_mel_pitch = [pitch for pitch in sub_mel_pitch if pitch > 0]  # Filter out zero pitches

        # Do a binary classification of it, to determine assign to high or low string
        voice_assign = [-1] * 8  # -1 means not assigned, 0 means low string, 1 means high string
        sub_mel_pitch_mean = sum(all_sub_mel_pitch) / len(all_sub_mel_pitch)
        for pos in filling_positions:
            if sub_mel_pitch[pos] > sub_mel_pitch_mean:
                voice_assign[pos] = 1
            else:
                voice_assign[pos] = 0
        voice_first_half = voice_assign[:4]
        voice_second_half = voice_assign[4:]
        bass_string_first_half = bass_strings_used_by_position[:4]
        bass_string_second_half = bass_strings_used_by_position[4:]

        # For each half of the bar, assign filling notes
        # Determine strings to use for filling notes based on voice_assign, bass_string_used_by_position, and melody_strings_used_by_position
        string_to_use = [-1] * 8  # (low_string, high_string) for each position
        for pos in filling_positions:
            mel_string = melody_strings_used_by_position[pos]
            bass_string = bass_strings_used_by_position[pos]
            usable_strings = set(range(1, 7))  # Strings 1 to 6
            # Deduct bass string and lower strings
            if bass_string > 0:
                usable_strings -= set(range(bass_string, 7))
            # Deduct melody string and higher strings
            if mel_string > 0:
                usable_strings -= set(range(1, mel_string + 1))
            n_usable_strings = len(usable_strings)
            if n_usable_strings >= 2:
                # If there are at least 2 usable strings, assign to the higher string for high voice and lower string for low voice
                if voice_assign[pos] == 1: # High voice
                    string_to_use[pos] = bass_string - 2 # Use 2 strings higher than bass string
                elif voice_assign[pos] == 0: # Low voice
                    string_to_use[pos] = bass_string - 1 # Use one string higher than bass string
            elif n_usable_strings == 1:
                # Use the only usable string as low string, and repeat melody string as high string
                if voice_assign[pos] == 1: # High voice
                    string_to_use[pos] = mel_string
                elif voice_assign[pos] == 0: # Low voice
                    string_to_use[pos] = bass_string - 1
            else:
                # No usable strings
                # Use the bass string as low string, and repeat melody string as high string
                if voice_assign[pos] == 1: # High voice
                    string_to_use[pos] = mel_string
                elif voice_assign[pos] == 0: # Low voice
                    string_to_use[pos] = bass_string

        sfs_1 = chart_1.string_fret_list
        sfs_2 = chart_2.string_fret_list
        # Fill the filling notes into the tab
        for pos in filling_positions:
            # Get the string to use for this position
            string_id = string_to_use[pos]

            # Get all useable sf positions from the chart
            if pos < 4:
                sfs = sfs_1
            else:
                sfs = sfs_2
            # Find the fret for this string in the chart
            sf_of_the_string = [sf for sf in sfs if sf[0] == string_id]
            if len(sf_of_the_string) == 0:
                # If no string-fret pair found, skip this position
                continue
            # If there are multiple string-fret pairs, randomly choose one
            sf_selected = random.choice(sf_of_the_string)
            psf = (pos, sf_selected[0], sf_selected[1])  # (position, string_id, fret)

            tab.add_note(psf[0], psf[1], psf[2])  # Add the note to the tab
            b = 2

        a = 2

        return tab
    
    def get_psf_from_chart(self, melody: 'NoteSeq', chart: 'Chart') -> list:
        """
        Find every position-string-fret pair for each melody note in the chart.

        Args:
            melody: NoteSeq object containing melody notes (each note must have .get_note_name()).
            chart: Chart object containing string-fret mapping for the bar.

        Returns:
            List of tuples (string_id, fret, note_name) for each melody note found in the chart.
        """
        result = []

        for melody_note in melody:
            pos = melody_note.onset // 6  # Convert to 8th note position
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
            a = 2
        tab_of_the_song = TabSeq(tab_of_the_bars)
        
        
        return tab_of_the_song


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


def get_filling_notes(note_seq: 'NoteSeq', melody_seq: 'NoteSeq', bass_seq: 'NoteSeq') -> 'NoteSeq':
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


def get_filling_note_density_by_position(filling_notes: 'NoteSeq') -> dict:
    """
    Calculate the filling note density for each onset position directly from filling_notes.

    Args:
        filling_notes: NoteSeq object containing only filling notes (each note must have .onset).

    Returns:
        Dictionary mapping onset position (int) to filling note density (int).
    """
    density_by_position = {}
    for note in filling_notes.notes:
        onset = note.onset // 6  # Convert to 8th note position if needed
        density_by_position[onset] = density_by_position.get(onset, 0) + 1
    return density_by_position


def get_filling_submelody_contour(filling_notes: 'NoteSeq') -> dict:
    """
    Calculate the filling sub-melody contour: for each onset position,
    find the highest note (by pitch) among filling notes.

    Args:
        filling_notes: NoteSeq object containing only filling notes (each note must have .onset and .pitch).

    Returns:
        Dictionary mapping onset position (int) to the highest filling note (Note object).
    """
    contour = {}
    onset_dict = {}
    for note in filling_notes.notes:
        onset = note.onset // 6  # Convert to 8th note position if needed
        onset_dict.setdefault(onset, []).append(note)
    for onset, notes_at_onset in onset_dict.items():
        highest_note = max(notes_at_onset, key=lambda n: n.pitch)
        contour[onset] = highest_note
    return contour


if __name__ == '__main__':
    main()