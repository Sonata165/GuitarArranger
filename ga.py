import os
import sys
from typing import List, Tuple
import random
from remi_z import MultiTrack, Bar, Note, detect_chord_from_pitch_list
import numpy as np
from sklearn.metrics import f1_score


def main():
    # Obtain the input bar from MIDI
    mt = MultiTrack.from_midi('/Users/sonata/Code/GuitarArranger/misc/caihong-4bar.midi')
    ref_bar = mt[0]

    # Get melody
    mel_notes = ref_bar.get_melody('hi_note')
    print(mel_notes)

    # Get chord
    chords = ref_bar.get_chord()
    print(chords)

    # Get notes
    all_notes = ref_bar.get_all_notes()
    all_notes = [(note.onset, note.pitch) for note in all_notes]

    # Run GA
    ga = GeneticAlgorithm(
        population_size=100, 
        num_positions=8, 
        mutation_rate=0.1,
        crossover_rate=0.7, 
        mel_notes=mel_notes,
        chords=chords,
        notes=all_notes,
    )
    best_individual, best_fitness = ga.run(num_generations=500)

    # Print results
    print("Best Individual:")
    print(best_individual)
    print("Best Fitness:", best_fitness)


def test_visualization():
    pos = Position([1, 2, 3, 4, 5, 6])
    print(pos)

    pos2 = Position([2, 3, '-', 5, 6, 7])
    pos_seq = PositionSeqBar([pos, pos2])
    print(pos_seq)


class Position:
    '''
    A fixed length list (len=6) representing fret number to be pressed on 6 strings
    from low to high
    '''
    def __init__(self, pos):
        '''
        Initialize a Position 
        '''
        assert isinstance(pos, list)
        assert len(pos) == 6, f"Position should have length 6, but got {len(pos)}"
        self.pos = pos

    @classmethod
    def zero_init(cls):
        '''
        Initialize a Position with all strings open
        '''
        return cls(['-' for _ in range(6)])
    
    def random_fill(self):
        '''
        Randomly fill the position with fret numbers in [0, 20] and '-'
        '-' has 90% chance
        [0, 20] share the rest 10% chance, smaller number has higher probability
        '''
        for i in range(6):
            if random.random() < 0.8:  # 90% chance for '-'
                self.pos[i] = '-'
            else:
                # 10% chance for a number between 0 and 20
                # Weighted random choice, smaller numbers have higher probability
                weights = [1.0 / (j + 1) for j in range(20)]  # Weights for 0 to 19
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                # Use choices for weighted random selection
                self.pos[i] = random.choices(range(20), weights=normalized_weights)[0]

    def get_pot_mel(self):
        '''
        Get potential melody
        I.e, highest note of this position
        '''
        strings = [6, 5, 4, 3, 2, 1]
        mel_string, mel_fret = None, None
        for i, fret in enumerate(self.pos):
            if fret != '-':
                mel_string = strings[i]
                mel_fret = fret

        mel_pitch = guitar_fret_to_midi(mel_string, mel_fret)
        
        return mel_pitch

    def __str__(self):
        return str(self.pos)
    
    def __repr__(self):
        return self.__str__()


class PositionSeqBar:
    '''
    A sequence of positions

    There will be 3 types of length in future:
    - len=8: 8-th note quantization
    - len=12: 12-th note quantization
    - len=16: 16-th note quantization
    '''
    def __init__(self, pos_seq:List[Position]):
        '''
        Initialize a PositionSeq from a list of Position
        '''
        assert isinstance(pos_seq, list)
        self.pos_seq = pos_seq

    @classmethod
    def zero_init(cls, num_positions:int):
        '''
        Zero initialize a PositionSeq with num_positions
        '''
        return cls([Position.zero_init() for _ in range(num_positions)])
    
    def random_fill(self):
        '''
        Randomly fill the position sequence with random fret numbers in [0, 20] and '-'
        '''
        for pos in self.pos_seq:
            pos.random_fill()

    def get_pot_mel(self):
        '''
        Get potential melody
        I.e, highest note of this position sequence
        '''
        mel_pitches = [pos.get_pot_mel() for pos in self.pos_seq]
        return mel_pitches
    
    def get_all_notes(self):
        pos_seq = self.pos_seq
        all_notes = []
        for j, pos in enumerate(pos_seq):
            for i, fret in enumerate(pos.pos):
                if fret != '-':
                    pitch = guitar_fret_to_midi(6 - i, fret)
                    all_notes.append((j, pitch))
        return all_notes

    def calculate_mel_dif(self, mel_notes:List[Note]):
        '''
        mel_notes: melody notes of this bar, in 48-th note quantization
        '''
        # Get potential melody of this bar
        pot_mel = self.get_pot_mel()

        # Get required melody
        mel_pitches = [note.pitch for note in mel_notes]

        # TODO: mel_notes may not use None to represent silence, need fix
        mel_diff = []
        for pmel, mel in zip(pot_mel, mel_pitches):
            if mel is None:
                pass
            else:
                if pmel is not None:
                    mel_diff.append(abs(pmel - mel))
                else:
                    mel_diff.append(mel)

        ret = sum(mel_diff) / len(mel_diff)

        return ret

    def calculate_position_penalty(self):
        '''
        Calculate position (把位) penalty
        '''
        penalty = []
        for pos_tup in self.pos_seq:
            for t in pos_tup.pos:
                if t != '-':
                    penalty.append(t)
        if len(penalty) == 0:
            return 0
        penalty = sum(penalty) / len(penalty)
        return penalty
    
    def calculate_intra_position_penalty(self):
        '''
        Calculate fitness within each position
        '''
        penalty = []
        for pos_list in self.pos_seq:
            max_fret = -1
            min_fret = 21
            for t in pos_list.pos:
                if t != '-':
                    max_fret = max(max_fret, t)
                    min_fret = min(min_fret, t)
            
            fret_range = max(max_fret - min_fret, 0)

            ''' Convert fret range to penalty 
            0~2: 0
            3: 1
            4: 200
            > 4: 999
            '''
            if fret_range <= 2:
                t = 0
            elif fret_range == 3:
                t = 1
            elif fret_range == 4:
                t = 200
            else:
                t = 999
            penalty.append(t)
        if len(penalty) == 0:
            return 0
        penalty = sum(penalty) / len(penalty)
        return penalty
        
    def calculate_chord_dif(self, ref_chords):
        ''' Determine the chord of this object '''
        # Convert this object to two pitch sequence
        all_notes = self.get_all_notes()

        # Get chord from notes
        pitch_seq_1 = [note[1] for note in all_notes if note[0] < 4]
        pitch_seq_2 = [note[1] for note in all_notes if note[0] >= 4]
        chord_1 = detect_chord_from_pitch_list(pitch_seq_1, return_root_name=True)
        chord_2 = detect_chord_from_pitch_list(pitch_seq_2, return_root_name=True)
        # print(chord_1, chord_2)

        # Calculate difference
        dif = 0
        if ref_chords[0] is not None:
            if chord_1 is not None:
                if chord_1[0] != ref_chords[0][0]:
                    dif += 0.5
                if chord_1[1] != ref_chords[0][1]:
                    dif += 0.5
        if ref_chords[1] is not None:
            if chord_2 is not None:
                if chord_2[0] != ref_chords[1][0]:
                    dif += 0.5
                if chord_2[1] != ref_chords[1][1]:
                    dif += 0.5
        dif /= 2
        
        return dif
    
    def calculate_note_dif(self, ref_notes):
        ''' Determine the chord of this object '''
        # Get all notes from output
        all_notes = self.get_all_notes()

        # Calculate F1
        f1 = calculate_note_recall(all_notes, ref_notes)

        # Convert to min mode
        dif = 1 - f1

        return dif


    def __str__(self):
        if not self.pos_seq:  # 处理空序列的情况
            return ""

        num_positions = len(self.pos_seq)
        lines = [["-" for _ in range(num_positions)] for _ in range(6)]
        strings = ["E", "B", "G", "D", "A", "E"]
        output = ""

        for i, position in enumerate(self.pos_seq):
            for j, fret in enumerate(position.pos):
                # lines[5 - j][i] = str(fret)  # 注意这里使用 5 - j 来正确对应弦
                if fret != '-':
                    lines[5 - j][i] = f"{fret:<2d}"
                else:
                    lines[5 - j][i] = '- '

        for j in range(6):
            output += strings[j] + "|" + "".join(lines[j]) + "|\n"

        return output
    
    def __repr__(self):
        return self.__str__()
    

class BarSeq:
    '''
    A sequence of bars
    '''
    def __init__(self, bars:List[PositionSeqBar]):
        '''
        Initialize a BarSeq from a list of Bar
        '''
        assert isinstance(bars, list)
        self.bars = bars

    @classmethod
    def zero_init(cls, num_bars:int, num_positions:int):
        '''
        Zero initialize a BarSeq with num_bars
        '''
        return cls([PositionSeqBar.zero_init(num_positions) for _ in range(num_bars)])
    
    def random_fill(self):
        '''
        Randomly fill the bar sequence with random fret numbers in [0, 20] and '-'
        '''
        for bar in self.bars:
            bar.random_fill()

    def calculate_mel_dif(self, mel_notes_bars:List[List[Note]]):
        res = []
        for bar, mel_notes in zip(self.bars, mel_notes_bars):
            t = bar.calculate_mel_dif(mel_notes)
            res.append(t)
        ret = sum(res) / len(res)
        return ret

    def __str__(self):
        if not self.bars:
            return ""

        num_bars = len(self.bars)
        lines = [[] for _ in range(6)]  # 初始化空列表

        for bar in self.bars:
            bar_lines = str(bar).strip().split('\n')  # 获取每个bar的字符串表示并分割成行
            for i, line in enumerate(bar_lines):
                lines[i].extend(list(line[2:-1])) # 去除弦名和两侧的|

            #在小节之间添加分隔符，除了最后一个小节
            if bar != self.bars[-1]:
                for i in range(6):
                    lines[i].append("|")

        output = ""
        strings = ["E", "B", "G", "D", "A", "E"]
        for i in range(6):
            output += strings[i] + "|" + "".join(lines[i]) + "|\n"
        return output
    
    def __repr__(self):
        return self.__str__()


def guitar_fret_to_midi(string: int, fret: int) -> int:
    """
    将吉他弦号和品格数转换为 MIDI 音高 ID。

    吉他弦号：6 (E) 到 1 (e)
    品格数：0 表示空弦

    如果输入参数为 None 或超出范围，则返回 None。
    """

    if string is None or fret is None:
        return None

    if not isinstance(string, int) or not isinstance(fret, int):
        return None

    if not 1 <= string <= 6 or fret < 0:
        return None

    # 标准调弦下各弦的 MIDI 音高
    string_pitches = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, e4
    string_pitches = string_pitches[::-1]  # 反转，使得索引 0 对应 6 弦

    midi_pitch = string_pitches[string - 1] + fret
    return midi_pitch


class GeneticAlgorithm:
    '''
    下面是关于GA的一些想法：
    - 旋律
        - 严格一致
        - （放松的匹配）
    - 和弦
        - 最大匹配after melody ()
        - （计算和弦类型相同）
    - Texture
        - 低音线条变化趋势匹配
        - （note density考虑）
    - Playability
        - position内部
        - position之间
        - block的多个position之间
    '''
    def __init__(self, population_size, num_positions, mutation_rate, crossover_rate, mel_notes, chords, notes):
        self.population_size = population_size
        self.num_positions = num_positions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mel_notes = mel_notes
        self.population = [PositionSeqBar.zero_init(num_positions) for _ in range(population_size)]
        self.best_individual = None
        self.best_fitness = float('inf')
        self.chords = chords
        self.notes = notes

    def initialize_population(self):
        for individual in self.population:
            individual.random_fill()

    def calculate_fitness(self, individual:PositionSeqBar):
        mel_dif = individual.calculate_mel_dif(self.mel_notes) # 只计算第一个bar的fitness
        pos_pel = individual.calculate_position_penalty()
        intra_pos_pel = individual.calculate_intra_position_penalty()

        # Chord difference
        chord_dif = individual.calculate_chord_dif(self.chords)

        # Note difference
        note_dif = individual.calculate_note_dif(self.notes)

        fitness = mel_dif * 100 \
                + pos_pel * 1 \
                + intra_pos_pel \
                + chord_dif * 50 \
                + note_dif * 100
        # print(mel_dif)
        return fitness

    def selection(self):
        # Tournament selection
        tournament_size = 3
        winners = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=self.calculate_fitness)
            winners.append(winner)
        return winners

    def crossover(self, parent1: PositionSeqBar, parent2: PositionSeqBar) -> PositionSeqBar:
        """
        交叉操作：以一定的概率将两个父代 PositionSeqBar 对象的部分“基因”进行交换，生成一个新的子代 PositionSeqBar 对象。

        Args:
            parent1: 第一个父代 PositionSeqBar 对象。
            parent2: 第二个父代 PositionSeqBar 对象。

        Returns:
            一个新的子代 PositionSeqBar 对象，或者如果未进行交叉，则返回父代之一的副本。
        """
        if random.random() < self.crossover_rate:  # 以 crossover_rate 的概率进行交叉
            child = PositionSeqBar.zero_init(self.num_positions)  # 创建一个空的子代
            for i in range(self.num_positions):  # 遍历每个位置
                if random.random() < 0.5:  # 以 50% 的概率选择 parent1 的基因
                    child.pos_seq[i] = Position(parent1.pos_seq[i].pos[:])  # 深拷贝！避免修改原对象
                else:  # 否则选择 parent2 的基因
                    child.pos_seq[i] = Position(parent2.pos_seq[i].pos[:])  # 深拷贝！避免修改原对象
            return child  # 返回新的子代
        else:
            # 不进行交叉，返回 parent1 的一个副本，确保返回类型一致
            new_pos_seq = []
            for p in parent1.pos_seq:
                new_pos_seq.append(Position(p.pos[:]))
            return PositionSeqBar(new_pos_seq)

    def mutation(self, individual):
        for i in range(self.num_positions):

            # Position-based
            if random.random() < self.mutation_rate:
                individual.pos_seq[i].random_fill() # old

                # # Random select string to mutate
                # # First random choose a number from possion distribution
                # n_str = min(np.random.poisson(2), 6)
                # string_index = random.sample(range(6), k=n_str)
                # for j in string_index:
                    
                #     # Random choose candidate
                #     candidate = [i for i in range(20)] + ['-']
                #     t = random.choice(candidate)
                #     individual.pos_seq[i].pos[j] = t

            # # String-based
            # for j in range(6):
            #     if random.random() < self.mutation_rate:
            #         candidate = [i for i in range(20)] + ['-']
            #         t = random.choice(candidate)
            #         individual.pos_seq[i].pos[j] = t
            #         b = 2

    def run(self, num_generations):
        self.initialize_population()

        for generation in range(num_generations):
            fitnesses = [self.calculate_fitness(individual) for individual in self.population]
            best_in_gen_index = fitnesses.index(min(fitnesses))
            best_in_gen = self.population[best_in_gen_index]
            best_in_gen_fitness = fitnesses[best_in_gen_index]

            if best_in_gen_fitness < self.best_fitness:
                self.best_fitness = best_in_gen_fitness
                self.best_individual = best_in_gen

            if generation % 10 == 0:
                print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness:.03f}")

            new_population = []
            winners = self.selection()
            for i in range(0, self.population_size, 2):
                parent1 = winners[i]
                if i + 1 < self.population_size:
                    parent2 = winners[i + 1]
                    child = self.crossover(parent1, parent2)
                    if child is not None:
                        new_population.append(child)
                else:
                    new_population.append(Position(parent1.pos_seq[:]))
            
            for individual in new_population:
                self.mutation(individual)
            
            self.population = new_population
        return self.best_individual, self.best_fitness


def calculate_note_recall(out_notes, ref_notes):

    ref_notes = set(ref_notes)

    tot_recall = 0
    for note in out_notes:
        if note in ref_notes:
            tot_recall += 1
    recall_score = tot_recall / len(ref_notes) if len(ref_notes) > 0 else 1
    
    return recall_score


def calculate_note_f1(out_notes, ref_notes):
    out_proll = get_proll_from_seq_q16(out_notes)
    tgt_proll = get_proll_from_seq_q16(ref_notes)

    f1 = bar_level_note_f1_from_proll(out_proll, tgt_proll)
    return f1


def get_proll_from_seq_q16(seq):
    '''
    Convert the sequence to a piano roll with 16th note resolution
    '''
    n_pos = 16
    proll = np.zeros((n_pos, 128), dtype=int)
    
    # Quantize
    pos_q = 0
    for i, (onset, pitch) in enumerate(seq):
        # Ensure pos_q and pitch are within the range
        pos_q = min(max(0, onset), n_pos - 1)
        pitch = min(max(0, pitch), 127)
        proll[pos_q, pitch] = 1
    return proll


def bar_level_note_f1_from_proll(proll_out:np.ndarray, proll_ref:np.ndarray):
    '''
    Compute the note F1 for a segment-level piano arrangement
    '''
    # Binarize
    proll_ref = (proll_ref > 0).astype(int) # [16, 128]
    proll_out = (proll_out > 0).astype(int)

    # Flatten
    proll_ref = proll_ref.flatten()
    proll_out = proll_out.flatten()

    # Calculate the note F1
    f1 = f1_score(proll_ref, proll_out)
    return f1




if __name__ == '__main__':
    main()