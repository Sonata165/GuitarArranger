'''
这个文件用来描述Rule-Based Guitar Arranger的伪代码
'''

def main():
    pass

class ArrangerSystem:
    def do_the_job(midi):
        note_seq = midi_to_note_seq(midi)
        melody_of_bars = extract_melody(note_seq)
        chord_of_bars = extract_chord(note_seq)
        note_seq_of_bars = cut_to_bars(note_seq)
        
        voicer = Voicer()
        chart_seq = voicer.generate_chart_sequence_for_song(melody_of_bars, chord_of_bars)

        arpeggiator = Arpeggiator()
        tab = arpeggiator.arpeggiate_a_song(chart_seq, note_seq_of_bars)

        out_note_seq = convert_to_note_seq(tab)
        more_accurate_note_seq = duration_renderer(out_note_seq)

        out_midi = convert_note_seq_to_midi(more_accurate_note_seq)
        return out_midi


class Voicer:
    def generate_chart_candidate_for_block(self, melody_notes, chord):
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
        pass

    def generate_chart_sequence_for_song(melody_notes, chord_progression):
        '''
        Find the best chart sequence for the melody and chord progression of a song.
        That minimize the left-hand position-wise movement on neck

        Implemented by a shortest path algorithm.
        '''
        pass

class Arpeggiator:
    '''
    Takes in a sequence of chart, and a note sequence as texture reference,
    Generate tab
    '''
    def calculate_groove_for_a_bar(bar(maybe a note sequence)):
        '''
        groove is represented by 
        onset position of bass note
        onset position of melody note
        counter melody onset position (highest note of filling)
        filling note onset density of each position (except melody and bass onset)
        '''
        pass

    def arpeggiate_a_bar(self, chart_list_of_the_bar, bar):
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。

        '''
        groove = self.calculate_groove_for_a_bar(bar)
        # Generate an empty tab
        # Fill melody note to melody note position
        # Fill bass note to bass position
        
        # Add fills: 
        '''
        这个算法用于为吉他独奏编曲中的“填充声部”（filling）部分分配右手手指（或弦位），以最大程度还原原曲质感。

        首先，从去除主旋律和低音后的原曲 MIDI 中提取每个位置的填充音符，识别出在每个和弦块（chord block）中 
            filling 部分的最高音，并将这些最高音连成一个“filling melody contour”。根据该 contour 的平均音高判断其整体处于高音区还是低音区。

        接着，计算每个非主旋律、非低音位置上的填充音符密度，并取所有位置的中位数作为 density 阈值。

        最后，对于每个 filling 位置，若其密度高于中位数，则分配两个手指进行填充（即两个音）；否则仅分配一个音，
            并根据 filling melody contour 所在音区，选择靠近的高音弦或低音弦来放置音符。

        该策略在尽量还原原曲的同时，控制右手复杂度，并利用简单规则生成合理可演奏的填充纹理。
        '''
        return tab_of_the_bar
    
    def arpeggiate_a_song(self, chart_list_of_the_song, note_seq_of_the_song):
        # call arpeggiate_a_bar function to do the job for each bar.
        return tab_of_the_song


class DurationRenderer:
    '''
    Modify the duration of the system to make them sounds more natural and practical
    '''
    def modify_duration(note_seq)
        return note_seq