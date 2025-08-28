from tab_util import Chart, Tab, TabSeq
from fretboard_util import Fretboard
from remi_z import NoteSeq

def main():
    test_tab()
    # test_chart()

def procedures():
    test_chart()

def test_tab():
    tab = Tab()
    tab.add_note(5, 3, 7)
    print(tab)

    tab1 = Tab()
    tab2 = Tab()
    tab3 = Tab()
    tab4 = Tab()
    tab5 = Tab()
    t = TabSeq([tab1, tab2, tab3, tab4, tab5], tab_per_row=2)
    print(t)

def test_chart():
    # # C major open chord: frets 0, 1, 2, 3
    # chart = Chart(string_fret_list=[(5, 3), (4, 2), (3, 0), (2, 1), (1, 0)])
    # print(chart.avg_fret, chart.position)
    # print("C major open chord:")
    # print(chart)

    # # A major barre chord at 5th fret: frets 5, 7, 7, 6, 5, 5
    # chart = Chart(string_fret_list=[(6, 5), (5, 7), (4, 7), (3, 6), (2, 5), (1, 5)])
    # print("A major barre chord at 5th fret:")
    # print(chart)

    # # E minor open chord: frets 0, 2, 2, 0, 0, 0
    # chart = Chart(string_fret_list=[(6, 0), (5, 2), (4, 2), (3, 0), (2, 0), (1, 0)])
    # print("E minor open chord:")
    # print(chart)

    # C major with C4 C5 G4 E4
    fretboard = Fretboard()
    chord_note_sfs = fretboard.press_chord('C', 
        position=7, 
        string_press_once=False, 
        enforce_root=True,
        closed=False,
        force_highest_pitch='E3',
        )
    chart = Chart(string_fret_list=chord_note_sfs)
    melody = ['G4', 'C5', 'C4', 'E3']
    melody_note_sfs = fretboard.press_note_set(note_set=melody, lowest_fret=7)
    chart.fret_more_note(melody_note_sfs)
    print("C major with G4 C5 C4 E3:")
    print(chart)

    mel_sf = fretboard.press_note_set(note_set=['G3', 'G4', 'B3', 'E3', 'G3', 'E3', 'B3', 'E3'], lowest_fret=2)
    chart = Chart(string_fret_list=mel_sf)
    print(chart)
    # [G3 G4 B3 E3 G3 E3 B3 E3]
    print('\n')


    mel_sf = fretboard.press_note_set(note_set=['F#3', 'A3'], lowest_fret=7)
    chord_upper_pitch_limit = 54 # F#3
    
    chord_note_sfs = fretboard.press_chord('D', 
                                            position=6, 
                                            string_press_once=False, 
                                            enforce_root=True,
                                            closed=False,
                                            force_highest_pitch='F#3',
                                            )
    chart = Chart(string_fret_list=chord_note_sfs)
    chart.fret_more_note(mel_sf)
    print(chart)

    # # D major shape at 10th fret: frets 10, 12, 11, 10
    # chart = Chart(string_fret_list=[(2, 11), (3, 10), (4, 10), (5, 12)])
    # print("D major shape at 10th fret:")
    # print(chart)

    # # G major, only strings 3, 4, 5, 6 played
    # chart = Chart(string_fret_list=[(2, 0), (3, 0), (4, 0), (5, 3)])
    # print("G major (4 strings):")
    # print(chart)
    # print('Position:', chart.get_position())

    # # Custom note names
    # chart = Chart(string_fret_list=[(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)])
    # print("Custom note names (G major):")
    # print(chart)
    # print('Position:', chart.get_position())

    # # Dense chart
    # chart = Chart(string_fret_list=[(0, 3), (0, 5), (1, 2), (1, 4),(2, 0), (3, 0), (4, 0), (5, 3)])
    # print("A dense chart:")
    # print(chart)
    # print('Position:', chart.get_position())

    # chart = Chart(string_fret_list=[(0, 3), (0, 5), (1, 2), (1, 4),(2, 0), (3, 0), (4, 0), (5, 3)])
    # chart.display_note_name = False
    # print("A dense chart without note names:")
    # print(chart)
    # print('Position:', chart.get_position())
    # print('Avg fret', chart.get_avg_fret())

    # try:
    #     chart = Chart(string_fret_list=[(0, 1), (1, 7), (2, 0), (3, 3), (4, 4), (5, 8)])
    #     print(chart)
    # except ValueError as e:
    #     print(f"Expected error: {e}")

if __name__ == "__main__":
    main()