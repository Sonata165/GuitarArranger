from remi_z import NoteSeq, ChordSeq, midi_pitch_to_note_name, note_name_to_midi_pitch
from typing import List
from fretboard_util import Fretboard
from tab_util import Chart


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
            chord_str = block.chord.__str__() if block.chord is not None else 'N/A'
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
            chord_name = str(block.chord)
            chart = Chart(string_fret_list=chord_note_sfs, chord_name=chord_name)

            # Add the melody notes to the chart
            melody_str = ' '.join(melody)
            chart.fret_more_note(melody_note_sfs, melody_list=melody_str)
            ret.append(chart)

        # If no chart candidate, fall back to a melody-only chart so that the
        # arpeggiator can always find the melody notes in the returned chart.
        if len(ret) == 0:
            print(f'Warning: No chart candidate found for melody {block.melody} with chord {block.chord}. '
                  f'Falling back to melody-only chart.')
            median_position = sorted(possible_positions)[len(possible_positions) // 2]
            melody_note_sfs = self.fretboard.press_note_set(note_set=melody, lowest_fret=median_position)
            chord_name = str(block.chord) if block.chord is not None else 'N/A'
            melody_str = ' '.join(melody)
            fallback_chart = Chart(string_fret_list=melody_note_sfs, chord_name=chord_name,
                                   melody_list=melody_str)
            return [fallback_chart]

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
        if not chart_candidates_seq and not all(chart_candidates_seq):
            return []

        # # Remove empty chart candidate layers
        # chart_candidates_seq = [layer for layer in chart_candidates_seq if layer]

        # If any layer is empty, copy the previous layer
        for i in range(1, len(chart_candidates_seq)):
            if not chart_candidates_seq[i]:
                chart_candidates_seq[i] = chart_candidates_seq[i-1]

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
