import numpy as np
from typing import Optional, Tuple, List

class Fretboard:
    def __init__(self):
        """Initialize the fretboard with standard tuning (EADGBE)."""
        # Initialize the pitch matrix (6 strings x 21 frets)
        self.pitch_mat = np.zeros((6, 21), dtype=int)
        
        # Initialize the status matrix (6 strings x 21 frets)
        self.status_matrix = np.zeros((6, 21), dtype=bool)
        
        # Standard tuning (EADGBE)
        self.tuning = [64, 59, 55, 50, 45, 40]  # MIDI pitch numbers for E4, A3, D3, G3, B2, E2
        
        # Fill the pitch matrix
        for i in range(6):  # 6 strings
            for j in range(21):  # 21 frets
                self.pitch_mat[i, j] = self.tuning[i] + j
        
        # Note names without octave
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Standard tuning notes for each string (from high E to low E)
        self.string_notes = ['E', 'B', 'G', 'D', 'A', 'E']

        # Chord interval patterns (in semitones from root)
        self.chord_patterns = {
            '': [0, 4, 7],      # Major triad
            'm': [0, 3, 7],     # Minor triad
            'dim': [0, 3, 6],   # Diminished triad
            'aug': [0, 4, 8],   # Augmented triad
            'sus4': [0, 5, 7],  # Suspended 4th
            'sus2': [0, 2, 7],  # Suspended 2nd
            'M7': [0, 4, 7, 11],    # Major 7th
            'm7': [0, 3, 7, 10],    # Minor 7th
            '7': [0, 4, 7, 10],     # Dominant 7th
            'o': [0, 3, 6, 9],      # Diminished 7th
            'm7b5': [0, 3, 6, 10],  # Half-diminished 7th
        }

    def get_status(self, string, fret):
        return self.status_matrix[string, fret]

    def set_status(self, string, fret, status):
        self.status_matrix[string, fret] = status

    def get_note_class(self, string_id, fret):
        '''
        Get the note name for a given string ID and fret
        string_id: 1 (highest E) to 6 (lowest E)
        fret: 0 (open) to 20
        '''
        assert string_id in range(1, 7), f"Invalid string_id: {string_id}"
        
        # Convert string_id (1-6) to index (0-5)
        string_index = string_id - 1
        base_note = self.string_notes[string_index]
        base_index = self.note_names.index(base_note)
        new_index = (base_index + fret) % 12
        return self.note_names[new_index]
    
    def get_note_name(self, string_id: int, fret: int) -> str:
        """
        Get the note name with octave for a given string ID and fret.
        Returns a string like "C#4".

        Args:
            string_id: String ID (1-6, where 1 is highest E)
            fret: Fret number (0-20)

        Returns:
            Note name with octave (e.g., "C#4")
        """
        assert string_id in range(1, 7), f"Invalid string_id: {string_id}"
        pitch = self.get_pitch(string_id - 1, fret)
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        note_name = self.note_names[note_index]
        return f"{note_name}{octave}"

    def visualize(self, num_frets=21):
        '''
        Visualize the fretboard with note names
        If status_matrix is None, show all notes
        If status_matrix is provided, only show notes that are pressed (status=True)
        '''
        vis_matrix = []
        for string_id in range(1, 7):  # 1 to 6 (highest to lowest)
            row = []
            # 0th fret
            if self.status_matrix is None:
                note = self.get_note_class(string_id, 0)
                row.append(f"{note:^3}")  # Center align with ^3
            else:
                if self.get_status(string_id - 1, 0):
                    note = self.get_note_class(string_id, 0)
                    row.append(f"{note:^3}")  # Center align with ^3
                else:
                    row.append("   ")
            row.append("|")
            # 1st to num_frets-1 frets
            for fret in range(1, num_frets):
                if self.status_matrix is None:
                    note = self.get_note_class(string_id, fret)
                    row.append(f"{note:3}")
                else:
                    if self.get_status(string_id - 1, fret):
                        note = self.get_note_class(string_id, fret)
                        row.append(f"{note:3}")
                    else:
                        row.append("   ")
            vis_matrix.append(row)
        # Print
        strings = ['E', 'B', 'G', 'D', 'A', 'E']  # high E to low E
        output = ""
        for i, row in enumerate(vis_matrix):
            output += f"{strings[i]}|" + "".join(row) + "|\n"
        # Fret numbers
        output += f"  {'0':^3}|"  # Center align the 0 in 3-character space
        fret_numbers = []
        for i in range(1, num_frets):
            fret_numbers.append(f"{i:<3}")  # Left align other fret numbers
        output += "".join(fret_numbers)
        print(output)  # Print the output instead of returning it

    def get_pitch(self, string_id: int, fret: int) -> int:
        """Get the MIDI pitch at a specific string and fret position."""
        return self.pitch_mat[string_id, fret]
        
    def note_with_octave_to_pitch(self, note_with_octave: str) -> int:
        """Convert a note name with octave to MIDI pitch number.
        
        Args:
            note_with_octave: Note name with octave (e.g., "C4", "E#3", "Bb5")
            
        Returns:
            MIDI pitch number (e.g., 60 for C4)
            
        Raises:
            ValueError: If note format is invalid
        """
        if len(note_with_octave) < 2:
            raise ValueError("Note must include octave (e.g., 'C4')")
        
        # Extract the note name and octave
        note_name = note_with_octave[:-1]  # Everything except the last character
        octave = int(note_with_octave[-1])  # Last character is the octave
        
        # Validate the note name
        if note_name not in self.note_names:
            raise ValueError(f"Invalid note name: {note_name}")
        
        # Calculate the MIDI pitch
        # C4 is MIDI pitch 60
        base_pitch = 60  # C4
        note_index = self.note_names.index(note_name)
        octave_diff = octave - 4  # Difference from C4's octave
        midi_pitch = base_pitch + note_index + (octave_diff * 12)
        
        return midi_pitch

    def press_note(self, note: str, additive: bool = False) -> Optional[Tuple[int, int]]:
        """Press a specific note on the fretboard.
        
        Args:
            note: Note name with octave (e.g., "C4")
            additive: If True, add to existing positions without clearing
            
        Returns:
            Tuple of (string_id, fret) if found, None otherwise
        """
        return self.press_pitch(self.note_with_octave_to_pitch(note), additive)

    def show_pitch_mat(self, num_frets=21):
        '''
        Display the pitch matrix with highest string on top
        '''
        # Create the visualization matrix
        vis_matrix = []
        for string in range(5, -1, -1):  # From high E (5) to low E (0)
            row = []
            for fret in range(num_frets):
                pitch = self.pitch_mat[string, fret]
                row.append(f"{pitch:3}")
            vis_matrix.append(row)
        
        # Print the visualization
        strings = ['E', 'B', 'G', 'D', 'A', 'E']  # high E to low E
        output = ""
        
        # Print each string
        for i, row in enumerate(vis_matrix):
            output += f"{strings[i]}|" + "".join(row) + "|\n"
        
        # Print fret numbers
        output += f"  {'0':^3}|"  # Center align the 0
        fret_numbers = []
        for i in range(1, num_frets):
            fret_numbers.append(f"{i:<3}")  # Left align other fret numbers
        output += "".join(fret_numbers)
        
        return output

    def press_strings(self, fret_list):
        '''
        Press strings according to the given list of frets
        fret_list: 6-element list indicating fret to press for each string
                  [6th string fret, 5th string fret, 4th string fret, 3rd string fret, 2nd string fret, 1st string fret]
        Example: [0, 0, 2, 2, 1, 0] for an E major chord
        '''
        if len(fret_list) != 6:
            raise ValueError("fret_list must have exactly 6 elements")
        
        # Initialize status matrix if None
        if self.status_matrix is None:
            self.status_matrix = np.zeros((6, 21), dtype=bool)
        
        # Reset all strings to unpressed
        self.status_matrix.fill(False)
        
        # Press the specified strings
        # Convert from [6th, 5th, 4th, 3rd, 2nd, 1st] to [1st, 2nd, 3rd, 4th, 5th, 6th]
        for i, fret in enumerate(fret_list):
            if 0 <= fret <= 20:
                # Convert from 6-based index to 0-based index, and reverse the order
                string_index = 5 - i
                self.status_matrix[string_index, fret] = True
            else:
                raise ValueError(f"Invalid fret value ({fret}) for string {6-i}. Fret must be 0-20.")

    def find_all_SF_of_chord_notes(self, chord_name: str, press: bool = False, position: int = -1) -> List[Tuple[int, int]]:
        """Find all positions for all notes in a chord.
        
        Args:
            chord_name: Name of the chord (e.g., "C", "Am", "G7")
            press: If True, update the status matrix with the found positions
            position: Starting fret position (-1 means no position constraint)
            
        Returns:
            List of (string_id, fret) tuples for all positions
        """
        # Get chord notes without octaves
        chord_notes = self.get_chord_notes(chord_name)
        
        # Initialize status matrix if press is True
        if press:
            if self.status_matrix is None:
                self.status_matrix = np.zeros((6, 21), dtype=bool)
            self.status_matrix.fill(False)  # Clear previous positions
        
        # Find all positions for each note across all octaves
        all_positions = []
        for note in chord_notes:
            positions = self.find_all_SF_of_note_name(note)
            all_positions.extend(positions)
        
        # If position is specified, filter positions to only include:
        # 1. Open strings (fret 0)
        # 2. Frets within the span starting at the specified position
        if position >= 0:
            filtered_positions = []
            for string_id, fret in all_positions:
                if fret == 0 or (position <= fret <= position + 3):  # 4-fret span
                    filtered_positions.append((string_id, fret))
            all_positions = filtered_positions
        
        # Update status matrix if press is True
        if press:
            for string_id, fret in all_positions:
                self.status_matrix[string_id-1][fret] = True
        
        return all_positions

    def find_all_SF_of_note_with_octave(self, note_with_octave: str, press: bool = False) -> List[Tuple[int, int]]:
        """Find all string-fret positions for a specific note with octave.
        
        Args:
            note_with_octave: The note name with octave (e.g., "C4", "E4", "G4")
            press: If True, update the status matrix to mark the found positions
            
        Returns:
            A list of (string_id, fret) tuples representing all possible positions
            for the note at the specified octave
        """
        if not note_with_octave:
            return []
            
        # Convert note to MIDI pitch
        pitch = self.note_with_octave_to_pitch(note_with_octave)
        
        # Find all positions for this specific pitch
        positions = []
        for string_id in range(1, 7):  # Use 1-based string indices
            for fret in range(21):
                if self.get_pitch(string_id - 1, fret) == pitch:  # Convert to 0-based for get_pitch
                    positions.append((string_id, fret))
        
        # Update status matrix if press is True
        if press:
            if self.status_matrix is None:
                self.status_matrix = np.zeros((6, 21), dtype=bool)
            self.status_matrix.fill(False)  # Clear previous positions
            for string_id, fret in positions:
                # Convert 1-based string_id to 0-based index for status matrix
                self.status_matrix[string_id - 1, fret] = 1
        
        return positions

    def press_pitch(self, midi_pitch, additive=False, all=False):
        '''
        Press a specific MIDI pitch on the fretboard.

        This method takes a MIDI pitch number and finds the position(s)
        on the fretboard that correspond to that exact pitch. It updates
        the status_matrix to press those position(s).

        Parameters
        ----------
        midi_pitch : int
            The MIDI pitch number to find (e.g., 60 for C4)
        additive : bool, optional
            If True, adds the new position(s) to existing ones without clearing previous status.
            If False (default), clears all previous positions before adding new one(s).
        all : bool, optional
            If True, presses all positions where this pitch can be played.
            If False (default), presses only the first position found.

        Returns
        -------
        positions : list of tuple
            A list of (string_id, fret) tuples indicating the positions where the pitch is found.
            If all=False, the list will contain at most one position.
            If no positions are found, returns an empty list.

        Example
        -------
        >>> fretboard = Fretboard()
        >>> fretboard.press_pitch(60)  # C4
        [(1, 8)]
        >>> fretboard.press_pitch(60, all=True)  # All positions of C4
        [(1, 8), (2, 3), (3, 10), ...]
        '''
        # Find all positions on the fretboard that match this MIDI pitch
        positions = []
        for string_id in range(1, 7):  # 1 to 6 (highest to lowest)
            for fret in range(21):
                if self.pitch_mat[string_id - 1, fret] == midi_pitch:
                    positions.append((string_id, fret))
                    if not all:
                        break
            if not all and positions:
                break

        if not positions:
            return []

        # Press the found position(s)
        if self.status_matrix is None:
            self.status_matrix = np.zeros((6, 21), dtype=bool)
        if not additive:
            self.status_matrix.fill(False)
        for string_id, fret in positions:
            string_index = string_id - 1
            self.status_matrix[string_index, fret] = True

        return positions

    def press_note_set(self, note_set: List[str], lowest_fret: int) -> List[Tuple[int, int]]:
        """Press a set of notes at a specific fret position.
        
        Args:
            note_set: List of note names with octaves (e.g., ["C4", "E4", "G4"])
            lowest_fret: The lowest fret number to start from
            
        Returns:
            A list of (string_id, fret) tuples representing the positions where the notes are pressed
        """
        # Deduplicate note set
        note_set = list(set(note_set))

        # Initialize status matrix
        if self.status_matrix is None:
            self.status_matrix = np.zeros((6, 21), dtype=bool)
        self.status_matrix.fill(False)  # Clear previous positions
        
        # Get all positions for each note with correct octave
        all_positions = []
        for note in note_set:
            # Find all positions for this specific note with octave
            positions = []
            for string_id in range(1, 7):  # Use 1-based string indices
                for fret in range(21):
                    if self.get_note_with_octave(string_id, fret) == note:
                        positions.append((string_id, fret))
            all_positions.append(positions)
        
        # Find the best positions for each note within the fret span
        selected_positions = []
        for note_positions in all_positions:
            # Filter positions to be within the fret span
            valid_positions = []
            for string_id, fret in note_positions:
                if fret == 0 or (lowest_fret <= fret <= lowest_fret + 3):  # Allow open strings and positions within 4-fret span
                    valid_positions.append((string_id, fret))
            
            if not valid_positions:
                print(f"Warning: No valid position found for note {note} at fret {lowest_fret}")
                continue
            
            # Prefer positions on higher strings (lower string_id)
            best_position = min(valid_positions, key=lambda x: x[0])
            selected_positions.append(best_position)
            
            # Update status matrix
            string_id, fret = best_position
            self.status_matrix[string_id - 1, fret] = True
        
        return selected_positions

    def get_note_with_octave(self, string_id: int, fret: int) -> str:
        """Get the note name with octave for a given string ID and fret.
        
        Args:
            string_id: String ID (1-6, where 1 is highest E)
            fret: Fret number (0-20)
            
        Returns:
            Note name with octave (e.g., "C4", "E4", "G4")
        """
        assert string_id in range(1, 7), f"Invalid string_id: {string_id}"
        
        # Get the MIDI pitch
        pitch = self.get_pitch(string_id - 1, fret)  # Convert to 0-based for get_pitch
        
        # Convert MIDI pitch to note with octave
        # C4 is MIDI pitch 60
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        note_name = self.note_names[note_index]
        
        return f"{note_name}{octave}"

    def find_all_playable_position_of_note_set(self, note_set: List[str], max_fret_span: int = 4) -> List[int]:
        """Find all possible left-hand positions to play a set of notes.
        
        Args:
            note_set: List of note names with octaves (e.g., ["C4", "E4", "G4"])
            max_fret_span: Maximum number of frets that can be covered by the left hand (default: 4)
            
        Returns:
            A list of lowest fret numbers for all playable positions
        """
        # Get all positions for all notes
        all_positions = self.find_all_SF_of_note_set(note_set)
        
        # Check each possible starting fret (from 1 to 20 - max_fret_span + 1)
        playable_positions = []
        for start_fret in range(1, 21 - max_fret_span + 1):
            # Filter positions to only include:
            # 1. Open strings (fret 0)
            # 2. Frets within the span
            valid_positions = []
            for string_id, fret in all_positions:
                if fret == 0 or (start_fret <= fret <= start_fret + max_fret_span - 1):
                    valid_positions.append((string_id, fret))
            
            # Get the notes that can be played with these positions
            playable_notes = set()
            for string_id, fret in valid_positions:
                note = self.get_note_with_octave(string_id, fret)
                playable_notes.add(note)
            
            # Check if we can play all required notes
            if all(note in playable_notes for note in note_set):
                playable_positions.append(start_fret)
        
        return playable_positions

    def find_all_SF_of_note_name(self, note_name: str, press: bool = False) -> List[Tuple[int, int]]:
        """Find all string-fret positions for a given note name across all octaves.
        
        Args:
            note_name: Note name without octave (e.g., "C", "E#", "Bb")
            press: If True, updates the status matrix to press the found positions
            
        Returns:
            List of (string_id, fret) tuples indicating all positions where the note is found
            
        Example:
            >>> fretboard = Fretboard()
            >>> fretboard.find_all_SF_of_note_name("C")
            [(1, 8), (2, 3), (3, 10), ...]  # All positions of C in any octave
        """
        # Find all positions for this note name
        positions = []
        for string_id in range(1, 7):  # Changed from range(6) to range(1, 7)
            for fret in range(21):
                note = self.get_note_class(string_id, fret)
                if note == note_name:
                    positions.append((string_id, fret))
        
        # Press the found positions if requested
        if press:
            if self.status_matrix is None:
                self.status_matrix = np.zeros((6, 21), dtype=bool)
            self.status_matrix.fill(False)  # Always clear previous positions
            for string_id, fret in positions:
                self.status_matrix[string_id - 1, fret] = True  # Convert string_id to 0-based index
        
        return positions

    def find_all_SF_of_note_set(self, note_list: List[str], press: bool = False) -> List[Tuple[int, int]]:
        """Find all string-fret positions for a set of notes.
        
        Args:
            note_list: List of note names with octave (e.g., ["C4", "E4", "G4"])
            press: If True, updates the status matrix to press the found positions
            
        Returns:
            List of (string_id, fret) tuples indicating all positions where any of the notes are found
            
        Example:
            >>> fretboard = Fretboard()
            >>> fretboard.find_all_SF_of_note_set(["C4", "E4", "G4"])  # C major triad
            [(1, 8), (2, 3), (3, 10), ...]  # All positions of any note in the set
        """
        # Convert note list to set of MIDI pitches to handle duplicates
        pitches = set(self.note_with_octave_to_pitch(note) for note in note_list)
        
        # Find all positions on the fretboard that match any of these MIDI pitches
        positions = []
        for string_id in range(1, 7):  # Use 1-based string indices
            for fret in range(21):
                if self.get_pitch(string_id - 1, fret) in pitches:  # Convert to 0-based for get_pitch
                    positions.append((string_id, fret))
        
        # Press the found positions if requested
        if press:
            if self.status_matrix is None:
                self.status_matrix = np.zeros((6, 21), dtype=bool)
            self.status_matrix.fill(False)  # Always clear previous positions
            for string_id, fret in positions:
                self.status_matrix[string_id - 1, fret] = True  # Convert to 0-based for status matrix
        
        return positions

    def get_chord_notes(self, chord_name: str) -> List[str]:
        """Convert a chord name (e.g., "C", "Am", "G7", etc.) to a list of note names without octaves.
        
        Args:
            chord_name: The chord name (e.g., "C", "Am", "G7", "CM7", "Bdim", etc.)
            
        Returns:
            A list of note names without octaves (e.g., ["C", "E", "G"] for C major)
            
        Example:
            >>> fretboard = Fretboard()
            >>> fretboard.get_chord_notes("C")
            ['C', 'E', 'G']
            >>> fretboard.get_chord_notes("Am")
            ['A', 'C', 'E']
            >>> fretboard.get_chord_notes("G7")
            ['G', 'B', 'D', 'F']
        """
        if chord_name == 'N/A':
            return []

        # Convert flat notes to sharp notes
        flat_to_sharp = {
            'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#'
        }
        
        # Check if the chord name starts with a flat note
        for flat, sharp in flat_to_sharp.items():
            if chord_name.startswith(flat):
                chord_name = sharp + chord_name[len(flat):]
                break
        
        # Check for major seventh chord
        if chord_name.endswith("M7") or chord_name.endswith("maj7"):
            root = chord_name[:-2] if chord_name.endswith("M7") else chord_name[:-4]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 4) % 12],  # Major third
                self.note_names[(root_index + 7) % 12],  # Perfect fifth
                self.note_names[(root_index + 11) % 12]  # Major seventh
            ]
            
        # Handle minor seventh chords (e.g., "Am7", "Amin7")
        elif chord_name.endswith("m7") or chord_name.endswith("min7"):
            root = chord_name[:-2] if chord_name.endswith("m7") else chord_name[:-4]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 3) % 12],  # Minor third
                self.note_names[(root_index + 7) % 12],  # Perfect fifth
                self.note_names[(root_index + 10) % 12]  # Minor seventh
            ]
            
        # Handle half-diminished chords (e.g., "Bm7b5", "Bø7")
        elif chord_name.endswith("m7b5") or chord_name.endswith("ø7"):
            root = chord_name[:-4] if chord_name.endswith("m7b5") else chord_name[:-2]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 3) % 12],  # Minor third
                self.note_names[(root_index + 6) % 12],  # Diminished fifth
                self.note_names[(root_index + 10) % 12]  # Minor seventh
            ]
            
        # Handle suspended fourth chords (e.g., "Csus4")
        elif chord_name.endswith("sus4"):
            root = chord_name[:-4]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 5) % 12],  # Perfect fourth
                self.note_names[(root_index + 7) % 12]   # Perfect fifth
            ]
            
        # Handle suspended second chords (e.g., "Csus2")
        elif chord_name.endswith("sus2"):
            root = chord_name[:-4]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 2) % 12],  # Major second
                self.note_names[(root_index + 7) % 12]   # Perfect fifth
            ]
            
        # Handle diminished chords (e.g., "Bdim", "Bo")
        elif chord_name.endswith("dim") or chord_name.endswith("o"):
            root = chord_name[:-3] if chord_name.endswith("dim") else chord_name[:-1]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 3) % 12],  # Minor third
                self.note_names[(root_index + 6) % 12]   # Diminished fifth
            ]
            
        # Handle dominant seventh chords (e.g., "G7")
        elif chord_name.endswith("7"):
            root = chord_name[:-1]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 4) % 12],  # Major third
                self.note_names[(root_index + 7) % 12],  # Perfect fifth
                self.note_names[(root_index + 10) % 12]  # Minor seventh
            ]
            
        # Handle minor chords (e.g., "Am", "Amin")
        elif chord_name.endswith("m") or chord_name.endswith("min"):
            root = chord_name[:-1] if chord_name.endswith("m") else chord_name[:-3]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 3) % 12],  # Minor third
                self.note_names[(root_index + 7) % 12]   # Perfect fifth
            ]
            
        # Handle major chords (e.g., "C", "CM", "Cmaj")
        elif chord_name.endswith("M") or chord_name.endswith("maj"):
            root = chord_name[:-1] if chord_name.endswith("M") else chord_name[:-3]
            root_index = self.note_names.index(root)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 4) % 12],  # Major third
                self.note_names[(root_index + 7) % 12]   # Perfect fifth
            ]
            
        # Default to major triad
        else:
            root_index = self.note_names.index(chord_name)
            return [
                self.note_names[root_index],
                self.note_names[(root_index + 4) % 12],  # Major third
                self.note_names[(root_index + 7) % 12]   # Perfect fifth
            ]

    def press_chord(self, 
                    chord_name: str, 
                    position: Optional[int] = None, 
                    string_press_once: bool = True, 
                    enforce_root: bool = True, 
                    closed: bool = False,
                    force_highest_pitch: Optional[str] = None,
                    force_highest_string: Optional[int] = None
                    ) -> List[Tuple[int, int]]:
        """Press a chord at a specific position.
        
        Args:
            chord_name: Name of the chord (e.g., "C", "Am", "G7")
            position: Starting fret position (None means lowest possible position)
            string_press_once: If True, each string will be pressed at most once, choosing the lowest fret
            enforce_root: If True, ensure the lowest pressed string plays the root note
            closed: If True, do not use any open strings (0-fret notes)
            force_highest_pitch: If set, do not press any chord note whose pitch is strictly larger than this note (e.g., 'C#3')
            force_highest_string: If set, do not press any chord note whose string number is strictly larger than this number
            
        Returns:
            List of (string_id, fret) tuples for the pressed positions
        """
        # Initialize status matrix
        if self.status_matrix is None:
            self.status_matrix = np.zeros((6, 21), dtype=bool)
        self.status_matrix.fill(False)  # Clear previous positions
        
        # Get all playable positions for the chord
        playable_positions = self.find_playable_position_of_chord(chord_name)
        
        if not playable_positions:
            raise ValueError(f"No playable positions found for chord {chord_name}")
        
        # If position is specified, verify it's playable
        if position is not None:
            if position not in playable_positions:
                raise ValueError(f"Position {position} is not playable for chord {chord_name}. Playable positions: {playable_positions}")
            target_position = position
        else:
            # Use the lowest playable position
            target_position = min(playable_positions)
        
        # Get all possible positions for the chord at the target position
        all_positions = self.find_all_SF_of_chord_notes(chord_name, press=False, position=target_position)
        
        # Apply force_highest_pitch filter if set
        if force_highest_pitch is not None:
            max_pitch = self.note_with_octave_to_pitch(force_highest_pitch)
            filtered_positions = []
            for string_id, fret in all_positions:
                pitch = self.get_pitch(string_id - 1, fret)
                if pitch <= max_pitch:
                    filtered_positions.append((string_id, fret))
            all_positions = filtered_positions
            
        # Apply force_highest_string filter if set
        if force_highest_string is not None:
            filtered_positions = []
            for string_id, fret in all_positions:
                if string_id <= force_highest_string:
                    filtered_positions.append((string_id, fret))
                    continue
            all_positions = filtered_positions
        
        if not string_press_once:
            if enforce_root:
                root_note = chord_name[0]
                if len(chord_name) > 1 and chord_name[1] == '#':
                    root_note += '#'
                # Sort all_positions by string (6->1), then by fret (low->high)
                sorted_positions = sorted(
                    [p for p in all_positions if not closed or p[1] > 0],
                    key=lambda x: (6 - x[0], x[1])
                )
                pressed_positions = []
                found_root = False
                for string_id, fret in sorted_positions:
                    note = self.get_note_class(string_id, fret)
                    if not found_root:
                        if note == root_note:
                            found_root = True
                            self.status_matrix[string_id-1][fret] = True
                            pressed_positions.append((string_id, fret))
                        # else: skip this position
                    else:
                        self.status_matrix[string_id-1][fret] = True
                        pressed_positions.append((string_id, fret))
                return pressed_positions
            else:
                # Original behavior
                for string_id, fret in all_positions:
                    if not closed or fret > 0:  # Skip open strings if closed is True
                        self.status_matrix[string_id-1][fret] = True
                return all_positions
        else:
            # For each string, select the position with the lowest fret
            selected_positions = []
            string_positions = {}  # Map string_id to list of (string_id, fret) tuples
            
            # Group positions by string
            for string_id, fret in all_positions:
                if not closed or fret > 0:  # Skip open strings if closed is True
                    if string_id not in string_positions:
                        string_positions[string_id] = []
                    string_positions[string_id].append((string_id, fret))
            
            # Get root note of the chord
            root_note = chord_name[0]  # First character is the root note
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_note += '#'  # Handle sharp notes
            
            # For each string, select the position with the lowest fret
            for string_id in range(6, 0, -1):  # From lowest to highest string
                if string_id in string_positions:
                    positions = sorted(string_positions[string_id], key=lambda x: x[1])
                    
                    if enforce_root and not selected_positions:  # Only for the lowest string with positions
                        # Try to find root note position
                        root_positions = [pos for pos in positions if self.get_note_class(pos[0], pos[1]) == root_note]
                        if root_positions:
                            selected_positions.append(root_positions[0])  # Take the lowest fret for root note
                        continue  # Skip this string if no root note found
                    
                    # If not enforcing root or not the lowest string, take the lowest fret
                    selected_positions.append(positions[0])
        
        # Press the selected positions
        for string_id, fret in selected_positions:
            self.status_matrix[string_id-1][fret] = True
        
        return selected_positions

    def find_playable_position_of_chord(self, chord_name: str) -> List[int]:
        """Find all playable positions for a chord.
        
        Args:
            chord_name: Name of the chord (e.g., "C", "Am", "G7")
            
        Returns:
            List of fret numbers where the chord can be played
        """
        # Get chord notes without octaves
        chord_notes = self.get_chord_notes(chord_name)
        
        # Get all positions for each note
        all_positions = []
        for note in chord_notes:
            positions = self.find_all_SF_of_note_name(note)
            all_positions.extend(positions)
        
        # Check each possible starting fret (from 1 to 20 - max_fret_span + 1)
        playable_positions = []
        for start_fret in range(1, 21 - 3):  # 4-fret span
            # Filter positions to only include:
            # 1. Open strings (fret 0)
            # 2. Frets within the span starting at the specified position
            valid_positions = []
            for string_id, fret in all_positions:
                if fret == 0 or (start_fret <= fret <= start_fret + 3):  # 4-fret span
                    valid_positions.append((string_id, fret))
            
            # Get the notes that can be played with these positions
            playable_notes = set()
            for string_id, fret in valid_positions:
                note = self.get_note_class(string_id, fret)  # Use get_note_name to ignore octave
                playable_notes.add(note)
            
            # Check if we can play all required notes
            if all(note in playable_notes for note in chord_notes):
                playable_positions.append(start_fret)
        
        return playable_positions