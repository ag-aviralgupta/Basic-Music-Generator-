# =============================================================
#  PHASE 2 — MIDI Preprocessing for AI Music Generation
#  Run this AFTER installing: pip install music21 numpy
# =============================================================

import os
import pickle
import numpy as np
from music21 import corpus, converter, instrument, note, chord, stream

# -------------------------------------------------------------
# STEP 1: Load MIDI files from a folder
# -------------------------------------------------------------
# music21 can read any .mid file.
# For testing, we'll use a built-in Bach piece so you don't
# need to download anything yet.

def get_notes_from_midi(midi_path):
    """
    Parse one MIDI file and return a flat list of note/chord strings.

    A 'note' looks like:  'C4', 'G#3', 'B5'
    A 'chord' looks like: 'C4.E4.G4'  (notes joined by dots)
    """
    notes_list = []

    # Load the MIDI file into music21
    midi = converter.parse(midi_path)

    # Separate each instrument track
    parts = instrument.partitionByInstrument(midi)

    # Use the first instrument part (usually piano/melody)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes  # fallback: all notes flat

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            # Single note → convert to string like "C4", "G#3"
            notes_list.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            # Chord → join all pitches with dots: "C4.E4.G4"
            notes_list.append('.'.join(str(n) for n in element.normalOrder))

    return notes_list


def load_all_midis(midi_folder):
    """
    Walk through a folder of .mid files and collect all notes.
    Returns one big flat list of note/chord strings.
    """
    all_notes = []

    midi_files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')]
    print(f"Found {len(midi_files)} MIDI files.")

    for filename in midi_files:
        path = os.path.join(midi_folder, filename)
        print(f"  Parsing: {filename}")
        notes = get_notes_from_midi(path)
        all_notes.extend(notes)
        print(f"    → {len(notes)} notes extracted")

    print(f"\nTotal notes collected: {len(all_notes)}")
    return all_notes


# -------------------------------------------------------------
# STEP 2: Build a vocabulary (note → integer mapping)
# -------------------------------------------------------------
# Think of this like a dictionary:
#   'C4'        → 0
#   'D4'        → 1
#   'E4'        → 2
#   'C4.E4.G4'  → 3
#   ...

def build_vocabulary(all_notes):
    """
    Create two lookup tables:
      note_to_int : {'C4': 0, 'D4': 1, ...}
      int_to_note : {0: 'C4', 1: 'D4', ...}
    """
    unique_notes = sorted(set(all_notes))
    vocab_size = len(unique_notes)

    note_to_int = {note: idx for idx, note in enumerate(unique_notes)}
    int_to_note = {idx: note for idx, note in enumerate(unique_notes)}

    print(f"Vocabulary size: {vocab_size} unique notes/chords")
    return note_to_int, int_to_note, vocab_size


# -------------------------------------------------------------
# STEP 3: Create training sequences (sliding window)
# -------------------------------------------------------------
# We slide a window of size SEQ_LEN across all notes.
# Each window = one training example.
#
#  Notes:  [C4, D4, E4, F4, G4, A4, B4, C5 ...]
#  Window 1: input=[C4,D4,E4,F4,G4]  → target=A4
#  Window 2: input=[D4,E4,F4,G4,A4]  → target=B4
#  Window 3: input=[E4,F4,G4,A4,B4]  → target=C5
#  ...

def create_sequences(all_notes, note_to_int, seq_len=50):
    """
    Returns:
      X : numpy array of shape (num_sequences, seq_len)
          — each row is SEQ_LEN integer note IDs
      y : numpy array of shape (num_sequences,)
          — each value is the next note ID (the label)
    """
    X = []  # inputs
    y = []  # targets (labels)

    encoded = [note_to_int[n] for n in all_notes]  # convert notes to ints

    for i in range(len(encoded) - seq_len):
        input_seq  = encoded[i : i + seq_len]       # 50 notes
        target_note = encoded[i + seq_len]           # the 51st note
        X.append(input_seq)
        y.append(target_note)

    X = np.array(X)  # shape: (num_sequences, seq_len)
    y = np.array(y)  # shape: (num_sequences,)

    print(f"Training sequences created: {len(X)}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    return X, y


# -------------------------------------------------------------
# STEP 4: Save everything to disk
# -------------------------------------------------------------
# We save X, y, and the vocabulary so Phase 3 can load them.

def save_preprocessed_data(X, y, note_to_int, int_to_note, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    with open(os.path.join(output_dir, 'note_to_int.pkl'), 'wb') as f:
        pickle.dump(note_to_int, f)

    with open(os.path.join(output_dir, 'int_to_note.pkl'), 'wb') as f:
        pickle.dump(int_to_note, f)

    print(f"\nSaved to '{output_dir}/' folder:")
    print(f"  X.npy         — input sequences")
    print(f"  y.npy         — target labels")
    print(f"  note_to_int.pkl — vocabulary (note → int)")
    print(f"  int_to_note.pkl — reverse vocabulary (int → note)")


# =============================================================
#  DEMO: Run with built-in Bach data (no download needed)
# =============================================================
# When you have real MIDI files, replace this section with:
#   all_notes = load_all_midis('your_midi_folder/')

if __name__ == '__main__':

    print("=" * 50)
    print("PHASE 2: Preprocessing MIDI Data")
    print("=" * 50)

    # --- Option A: Use built-in music21 Bach pieces (great for testing) ---
    print("\nLoading built-in Bach pieces for demo...")
    all_notes = []

    bach_paths = corpus.getComposer('bach')[:5]  # use 5 pieces
    for path in bach_paths:
        midi = corpus.parse(path)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                all_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                all_notes.append('.'.join(str(n) for n in element.normalOrder))

    print(f"Loaded {len(all_notes)} notes from Bach corpus.")

    # Build vocabulary
    print("\n--- Building Vocabulary ---")
    note_to_int, int_to_note, vocab_size = build_vocabulary(all_notes)

    # Show a sample of the vocabulary
    print("\nSample vocabulary (first 10 entries):")
    for note_str, idx in list(note_to_int.items())[:10]:
        print(f"  '{note_str}' → {idx}")

    # Create training sequences
    print("\n--- Creating Training Sequences ---")
    SEQ_LEN = 50  # how many notes the model sees at once
    X, y = create_sequences(all_notes, note_to_int, seq_len=SEQ_LEN)

    # Show a sample sequence
    print("\nSample training example (first sequence):")
    print(f"  Input  (first 5 of {SEQ_LEN}): {X[0][:5]} ...")
    print(f"  Target: {y[0]}  = '{int_to_note[y[0]]}'")

    # Save everything
    print("\n--- Saving Data ---")
    save_preprocessed_data(X, y, note_to_int, int_to_note, output_dir='data')

    print("\nDone! Phase 2 complete.")
    print(f"Vocabulary size : {vocab_size}")
    print(f"Training samples: {len(X)}")
    print(f"Sequence length : {SEQ_LEN}")
    print("\nNext step → Phase 3: Build the LSTM model.")
