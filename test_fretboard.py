from fretboard_util import Fretboard
import numpy as np


def main():
    """Run only the most recent test (get_chord_notes)"""
    # test_find_playable_position_of_chord()
    test_press_chord()
    # test_press_note_set()


def test_press_note():
    print("Testing press_note function:")
    print("C4 (standard mode):")
    fretboard = Fretboard()
    positions = fretboard.press_note("C4")
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nC4 (all positions):")
    fretboard = Fretboard()
    positions = fretboard.press_note("C4", all=True)
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nE4 (all positions):")
    fretboard = Fretboard()
    positions = fretboard.press_note("E4", all=True)
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nG4 (all positions, additive):")
    fretboard = Fretboard()
    positions = fretboard.press_note("G4", all=True, additive=True)
    print(f"Found at: {positions}")
    print(fretboard.visualize())

def test_press_pitch():
    print("\nTesting press_pitch function:")
    print("C4 (MIDI 60) - first position:")
    fretboard = Fretboard()
    positions = fretboard.press_pitch(60)
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nC4 (MIDI 60) - all positions:")
    fretboard = Fretboard()
    positions = fretboard.press_pitch(60, all=True)
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nE4 (MIDI 64) - all positions:")
    fretboard = Fretboard()
    positions = fretboard.press_pitch(64, all=True)
    print(f"Found at: {positions}")
    print(fretboard.visualize())

def test_find_all_frets_of_note():
    print("\nTesting find_all_frets_of_note function:")
    print("All positions of note C:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_with_octave("C")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nAll positions of note E (additive):")
    fretboard.find_all_SF_of_note_with_octave("E", additive=True)
    print(fretboard.visualize())
    
    print("\nAll positions of note G (additive):")
    fretboard.find_all_SF_of_note_with_octave("G", additive=True)
    print(fretboard.visualize())

def test_find_all_notes_in_chord():
    print("\nTesting find_all_notes_in_chord function:")
    print("C major chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("C")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nA minor chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Am")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nCsus2 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Csus2")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nCsus4 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Csus4")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nCM7 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("CM7")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nCm7 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Cm7")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nC7 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("C7")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nBo chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Bo")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    print("\nBm7b5 chord:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Bm7b5")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())

def test_find_all_SF_of_notes():
    print("Testing find_all_SF_of_notes function:")
    print("C4:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_notes("C4")
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nE4:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_notes("E4")
    print(f"Found at: {positions}")
    print(fretboard.visualize())
    
    print("\nG4:")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_notes("G4")
    print(f"Found at: {positions}")
    print(fretboard.visualize())

def test_find_all_SF_of_note_name():
    print("Testing find_all_SF_of_note_name function:")
    
    # Test without pressing
    print("\nAll positions of note C (without pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_name("C")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with pressing
    print("\nAll positions of note C (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_name("C", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a different note
    print("\nAll positions of note E (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_name("E", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())

def test_find_all_SF_of_note_set():
    print("Testing find_all_SF_of_note_set function:")
    
    # Test without pressing
    print("\nAll positions of C major triad [C4, E4, G4] (without pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_set(["C4", "E4", "G4"])
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with pressing
    print("\nAll positions of C major triad [C4, E4, G4] (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_set(["C4", "E4", "G4"], press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a more complex chord
    print("\nAll positions of C major seventh [C4, E4, G4, B4] (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_note_set(["C4", "E4", "G4", "B4"], press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())

def test_find_all_playable_position_of_note_set():
    print("Testing find_all_playable_position_of_note_set function:")
    
    fretboard = Fretboard()

    fretboard.find_all_SF_of_note_set(["C4", "E4", "G4"], press=True)
    print(fretboard.visualize())

    # Test with specific position
    print("\nC major triad at 3rd position:")
    t = fretboard.find_all_playable_position_of_note_set(["C4", "E4", "G4"])
    print(t)
    
    # Verify the expected playable positions
    expected_positions = [1, 3, 5, 8, 9, 10, 12, 14, 15, 17]
    assert t == expected_positions, f"Expected positions {expected_positions}, but got {t}"

    case2 = ["B4", "A4", "E5", "D5"]
    fretboard.find_all_SF_of_note_set(case2, press=True)
    print(fretboard.visualize())

    # Test with specific position
    print("\nC major triad at 3rd position:")
    t = fretboard.find_all_playable_position_of_note_set(case2)
    print(t)


def test_press_note_set():
    print("Testing press_note_set function:")
    
    fretboard = Fretboard()

    # Test with specific position
    print('All notes in C major triads in octave 4')
    case1 = ["C4", "E4", "G4"]
    fretboard.find_all_SF_of_note_set(case1, press=True)
    fretboard.visualize()

    print("C major triad at 3rd position:")
    t = fretboard.find_all_playable_position_of_note_set(case1)
    print('Possible positions:', t)

    print('Press with 1st position')
    fretboard.press_note_set(case1, 1)
    fretboard.visualize()

    print('Press with 3rd position')
    fretboard.press_note_set(case1, 3)
    fretboard.visualize()

    print('Press with 2nd position')
    fretboard.press_note_set(case1, 2)
    fretboard.visualize()
    
    # Test with specific melody
    print('All notes in a melody')
    case1 = ["B4", "E4", "E5", "D5", "A4"]
    fretboard.find_all_SF_of_note_set(case1, press=True)
    fretboard.visualize()
    t = fretboard.find_all_playable_position_of_note_set(case1)
    print('Possible positions:', t)
    fretboard.press_note_set(case1, 9)
    fretboard.visualize()

    fretboard.find_all_SF_of_chord_notes('CM7', press=True)
    fretboard.visualize()

    # Test with specific melody
    print('All notes in a melody')
    case1 = ["E4", "A3", "A4", "G4", "E4"]
    fretboard.find_all_SF_of_note_set(case1, press=True)
    fretboard.visualize()
    t = fretboard.find_all_playable_position_of_note_set(case1)
    print('Possible positions:', t)
    fretboard.press_note_set(case1, 2)
    fretboard.visualize()

    print('FM7 chord notes')
    fretboard.find_all_SF_of_chord_notes('FM7', press=True)
    fretboard.visualize()

    print('FM7, position 4')
    fretboard.find_all_SF_of_chord_notes('FM7', press=True, position=6)
    fretboard.visualize()

    print('G, position 1')
    fretboard.find_all_SF_of_chord_notes('C', press=True, position=1)
    fretboard.visualize()

    print('F#m7, position 4')
    fretboard.find_all_SF_of_chord_notes('F#', press=True)
    fretboard.visualize()

    print('New melody')
    t = fretboard.find_all_playable_position_of_note_set(['C#4', 'D#4', 'E4'])
    print(t)
    fretboard.press_note_set(['C#4', 'D#4', 'E4'], lowest_fret=8)
    fretboard.visualize()
    fretboard.find_all_SF_of_chord_notes('F#', press=True, position=8)
    fretboard.visualize()

def test_find_all_SF_of_chord_notes():
    print("Testing find_all_SF_of_chord_notes function:")
    
    # Test without pressing
    print("\nAll positions of C major chord (without pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("C")
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with pressing
    print("\nAll positions of C major chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("C", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a minor chord
    print("\nAll positions of A minor chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Am", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())

    # Test with C#7
    print("\nAll positions of C#7 chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("C#7", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a seventh chord
    print("\nAll positions of G7 chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("G7", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a major seventh chord
    print("\nAll positions of CM7 chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("CM7", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())
    
    # Test with a suspended chord
    print("\nAll positions of Csus4 chord (with pressing):")
    fretboard = Fretboard()
    positions = fretboard.find_all_SF_of_chord_notes("Csus4", press=True)
    print(f"Found {len(positions)} positions")
    print(fretboard.visualize())

def test_get_chord_notes():
    print("Testing get_chord_notes function:")
    
    fretboard = Fretboard()
    
    # Test major chords
    print("\nMajor chords:")
    print("C major:", fretboard.get_chord_notes("C"))  # Should be ["C4", "E4", "G4"]
    print("CM major:", fretboard.get_chord_notes("CM"))  # Should be ["C4", "E4", "G4"]
    print("Cmaj major:", fretboard.get_chord_notes("Cmaj"))  # Should be ["C4", "E4", "G4"]
    
    # Test minor chords
    print("\nMinor chords:")
    print("Am minor:", fretboard.get_chord_notes("Am"))  # Should be ["A4", "C4", "E4"]
    print("Amin minor:", fretboard.get_chord_notes("Amin"))  # Should be ["A4", "C4", "E4"]
    
    # Test seventh chords
    print("\nSeventh chords:")
    print("G7 dominant seventh:", fretboard.get_chord_notes("G7"))  # Should be ["G4", "B4", "D4", "F4"]
    print("CM7 major seventh:", fretboard.get_chord_notes("CM7"))  # Should be ["C4", "E4", "G4", "B4"]
    print("Am7 minor seventh:", fretboard.get_chord_notes("Am7"))  # Should be ["A4", "C4", "E4", "G4"]
    
    # Test suspended chords
    print("\nSuspended chords:")
    print("Csus4 suspended fourth:", fretboard.get_chord_notes("Csus4"))  # Should be ["C4", "F4", "G4"]
    print("Csus2 suspended second:", fretboard.get_chord_notes("Csus2"))  # Should be ["C4", "D4", "G4"]
    
    # Test diminished chords
    print("\nDiminished chords:")
    print("Bdim diminished:", fretboard.get_chord_notes("Bdim"))  # Should be ["B4", "D4", "F4"]
    print("Bo diminished:", fretboard.get_chord_notes("Bo"))  # Should be ["B4", "D4", "F4"]
    
    # Test half-diminished chords
    print("\nHalf-diminished chords:")
    print("Bm7b5 half-diminished:", fretboard.get_chord_notes("Bm7b5"))  # Should be ["B4", "D4", "F4", "A4"]
    print("Bø7 half-diminished:", fretboard.get_chord_notes("Bø7"))  # Should be ["B4", "D4", "F4", "A4"]
    
    # Test chords with accidentals
    print("\nChords with accidentals:")
    print("C# major:", fretboard.get_chord_notes("C#"))  # Should be ["C#4", "F4", "G#4"]
    print("Eb minor:", fretboard.get_chord_notes("Ebm"))  # Should be ["Eb4", "Gb4", "Bb4"]

def test_press_chord():
    """Test the press_chord function."""
    fretboard = Fretboard()
    
    fretboard.press_chord('C', 
                        position=1, 
                        string_press_once=False, 
                        enforce_root=True,
                        closed=False,
                        )
    fretboard.visualize()

    fretboard.press_chord('C', 
                        position=4, 
                        string_press_once=False, 
                        enforce_root=True,
                        closed=False,
                        )
    fretboard.visualize()

    # # Test pressing C chord at lowest position with string_press_once=True, enforce_root=True, closed=False
    # positions = fretboard.press_chord("C")
    # print("\nC chord at lowest position (string_press_once=True, enforce_root=True, closed=False):")
    # fretboard.visualize()
    
    # # Test pressing C chord at lowest position with string_press_once=True, enforce_root=True, closed=True
    # positions = fretboard.press_chord("C", closed=True)
    # print("\nC chord at lowest position (string_press_once=True, enforce_root=True, closed=True):")
    # fretboard.visualize()
    
    # # Test pressing C chord at position 3 with string_press_once=True, enforce_root=True, closed=False
    # positions = fretboard.press_chord("C", position=3)
    # print("\nC chord at position 3 (string_press_once=True, enforce_root=True, closed=False):")
    # fretboard.visualize()
    
    # # Test pressing C chord at position 4 with string_press_once=True, enforce_root=True, closed=True
    # positions = fretboard.press_chord("C", position=4, closed=True, enforce_root=False)
    # print("\nC chord at position 4 (string_press_once=True, enforce_root=True, closed=True):")
    # fretboard.visualize()
    
    # # Test pressing Am chord at position 5 with string_press_once=True, enforce_root=True, closed=True
    # positions = fretboard.press_chord("Am", position=5, closed=True)
    # print("\nAm chord at position 5 (string_press_once=True, enforce_root=True, closed=True):")
    # fretboard.visualize()
    
    # # Test pressing G7 chord at position 7 with string_press_once=True, enforce_root=True, closed=True
    # positions = fretboard.press_chord("G7", position=7, closed=True)
    # print("\nG7 chord at position 7 (string_press_once=True, enforce_root=True, closed=True):")
    # fretboard.visualize()
    
    # # Test pressing FM7 chord at position 1 with string_press_once=True, enforce_root=True, closed=True
    # positions = fretboard.press_chord("FM7", position=1, closed=True)
    # print("\nFM7 chord at position 1 (string_press_once=True, enforce_root=True, closed=True):")
    # fretboard.visualize()
    
    # # Test error case: invalid position
    # try:
    #     fretboard.press_chord("C", position=20)  # Position 20 is not playable
    #     assert False, "Should have raised ValueError"
    # except ValueError as e:
    #     print(f"\nExpected error: {e}")

def test_find_playable_position_of_chord():
    """Test the find_playable_position_of_chord function."""
    fretboard = Fretboard()
    
    # Test C chord
    positions = fretboard.find_playable_position_of_chord("C")
    print("\nC chord playable positions:", positions)
    
    # Test Am chord
    positions = fretboard.find_playable_position_of_chord("Am")
    print("\nAm chord playable positions:", positions)
    
    # Test G7 chord
    positions = fretboard.find_playable_position_of_chord("G7")
    print("\nG7 chord playable positions:", positions)
    
    # Test FM7 chord
    positions = fretboard.find_playable_position_of_chord("FM7")
    print("\nFM7 chord playable positions:", positions)
    
    # Visualize each chord at its first playable position
    for chord in ["C", "Am", "G7", "FM7"]:
        positions = fretboard.find_playable_position_of_chord(chord)
        if positions:
            print(f'All possible positions: {positions}')
            print(f"\n{chord} chord at position {positions[0]}:")
            fretboard.find_all_SF_of_chord_notes(chord, press=True, position=positions[0])
            fretboard.visualize()

def all_tests():
    """Run all test functions"""
    test_press_note()
    test_press_pitch()
    test_find_all_frets_of_note()
    test_find_all_notes_in_chord()
    test_find_all_SF_of_notes()
    test_find_all_SF_of_note_name()
    test_find_all_SF_of_note_set()
    test_find_all_playable_position_of_note_set()
    test_press_note_set()
    test_find_all_SF_of_chord_notes()
    test_get_chord_notes()
    test_press_chord()
    test_find_playable_position_of_chord()
    my_test()



def my_test():
    '''
    This is the author's test. Does not touch it in cursor or copilot.
    '''
    print('\nMy Test')
    fretboard = Fretboard()
    fretboard.find_all_SF_of_chord_notes("DM7")
    print(fretboard.visualize())

    fretboard.find_all_SF_of_chord_notes("E")
    print(fretboard.visualize())

    fretboard.find_all_SF_of_chord_notes("C#m7")
    print(fretboard.visualize())

    fretboard.find_all_SF_of_chord_notes("Fm7")
    print(fretboard.visualize())

    fretboard.press_strings([0, 5, 7, 6, 7, 0])
    print(fretboard.visualize())

if __name__ == "__main__":
    main()
