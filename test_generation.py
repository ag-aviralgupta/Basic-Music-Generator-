"""
Standalone Music Generation Test Script
Use this to test MIDI generation + audio conversion without the Streamlit UI
Helpful for debugging!
"""

import torch
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import music21
from music21 import instrument, note, chord, stream, tempo
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "model.pth"
VOCAB_PATH = DATA_DIR / "vocab.pkl"
NOTE_TO_INT_PATH = DATA_DIR / "note_to_int.pkl"
INT_TO_NOTE_PATH = DATA_DIR / "int_to_note.pkl"
OUTPUT_DIR = BASE_DIR / "generated_output"
SOUNDFONT_CANDIDATES = [
    DATA_DIR / "FluidR3_GM.sf2",
    DATA_DIR / "MuseScore_General.sf3",
    DATA_DIR / "MS Basic.sf3",
    Path(os.environ["SOUNDFONT_PATH"]) if os.environ.get("SOUNDFONT_PATH") else None,
    Path(r"C:\Program Files\MuseScore 4\sound\MS Basic.sf3"),
    Path(r"C:\Program Files\MuseScore 3\sound\MuseScore_General.sf3"),
    Path(r"C:\Program Files\MuseScore 2\sound\MuseScore_General.sf3"),
    Path(r"C:\Program Files (x86)\MuseScore 3\sound\MuseScore_General.sf3"),
    Path(r"C:\Program Files (x86)\MuseScore 2\sound\MuseScore_General.sf3"),
]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LSTM MODEL (Match your training code!)
# ============================================================================

class MusicLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=3, 
                                   batch_first=True, dropout=0.3)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_and_vocab():
    """Load trained model and vocabulary"""
    print("Loading model and vocabulary...")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return None, None, None

    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        note_to_int = vocab['note_to_int']
        int_to_note = vocab['int_to_note']
    elif NOTE_TO_INT_PATH.exists() and INT_TO_NOTE_PATH.exists():
        with open(NOTE_TO_INT_PATH, 'rb') as f:
            note_to_int = pickle.load(f)
        with open(INT_TO_NOTE_PATH, 'rb') as f:
            int_to_note = pickle.load(f)
    else:
        print("❌ Vocabulary not found.")
        print(f"   Expected {VOCAB_PATH}")
        print(f"   Or both {NOTE_TO_INT_PATH} and {INT_TO_NOTE_PATH}")
        return None, None, None

    vocab_size = len(note_to_int)
    
    print(f"✅ Vocabulary loaded: {vocab_size} unique notes")
    
    # Load model
    model = MusicLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, note_to_int, int_to_note


def generate_music(model, note_to_int, int_to_note, seed_note, num_notes=100, temperature=0.8):
    """Generate a sequence of notes"""
    print(f"\n🎵 Generating {num_notes} notes (seed: {seed_note}, temp: {temperature})...")
    
    model.eval()
    sequence = [note_to_int[seed_note]]
    hidden = None
    
    with torch.no_grad():
        for i in range(num_notes):
            input_tensor = torch.tensor([sequence[-1]], dtype=torch.long).unsqueeze(0)
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature
            logits = output[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Sample next note
            next_note_idx = np.random.choice(len(int_to_note), p=probabilities)
            sequence.append(next_note_idx)
            
            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/{num_notes} notes...")
    
    generated_notes = [int_to_note[idx] for idx in sequence]
    print(f"✅ Generation complete!")
    print(f"   First 20 notes: {generated_notes[:20]}")
    
    return generated_notes


def notes_to_midi(note_sequence, bpm=120, output_path=None):
    """Convert note sequence to MIDI"""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "generated_music.mid")
    
    print(f"\n🎹 Converting to MIDI ({bpm} BPM)...")
    
    s = stream.Score()
    part = stream.Part()
    part.append(instrument.Piano())
    part.append(tempo.MetronomeMark(number=bpm))
    
    valid_notes = 0
    for note_name in note_sequence:
        try:
            if note_name == 'rest':
                n = note.Rest(quarterLength=0.5)
            else:
                n = note.Note(note_name, quarterLength=0.5)
            part.append(n)
            valid_notes += 1
        except:
            continue
    
    s.append(part)
    s.write('midi', fp=output_path)
    
    print(f"✅ MIDI saved: {output_path}")
    print(f"   Valid notes: {valid_notes}/{len(note_sequence)}")
    
    return output_path


def find_soundfont():
    """Find a FluidSynth-compatible SoundFont file."""
    for soundfont in DATA_DIR.glob("*.sf[23]"):
        return soundfont

    for soundfont in SOUNDFONT_CANDIDATES:
        if soundfont and soundfont.exists():
            return soundfont

    return None


def midi_to_audio(midi_path, output_path=None, soundfont=None):
    """Convert MIDI to audio using FluidSynth"""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "generated_music.wav")
    
    print(f"\n🔊 Converting MIDI to audio...")
    
    if soundfont is None:
        soundfont = find_soundfont()

    if soundfont:
        print(f"   Found soundfont: {soundfont}")

    if not soundfont or not Path(soundfont).exists():
        print(f"⚠️  Soundfont not found. To get audio:")
        print(f"   1. Add any General MIDI .sf2 or .sf3 file to: {DATA_DIR}")
        print(f"   2. Or set SOUNDFONT_PATH to the full soundfont path")
        return None
    
    try:
        cmd = [
            'fluidsynth',
            '-ni',
            '-T', 'wav',
            '-F', output_path,
            '-r', '44100',
            str(soundfont),
            midi_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Audio saved: {output_path}")
        return output_path
    except FileNotFoundError:
        print(f"❌ FluidSynth not found!")
        print(f"   macOS: brew install fluidsynth")
        print(f"   Windows: choco install fluidsynth")
        print(f"   Linux: sudo apt-get install fluidsynth")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# TEST SCRIPT
# ============================================================================

def main():
    print("=" * 60)
    print("🎵 AI Music Generation - Test Script")
    print("=" * 60)
    
    # Load model
    model, note_to_int, int_to_note = load_model_and_vocab()
    if model is None:
        return
    
    # Test parameters
    seed_note = "C4"
    num_notes = 100
    temperature = 0.8
    bpm = 120
    
    print(f"\n📋 Test Configuration:")
    print(f"   Seed Note: {seed_note}")
    print(f"   Number of Notes: {num_notes}")
    print(f"   Temperature: {temperature}")
    print(f"   Tempo (BPM): {bpm}")
    
    # Generate music
    generated_notes = generate_music(
        model, note_to_int, int_to_note,
        seed_note=seed_note,
        num_notes=num_notes,
        temperature=temperature
    )
    
    # Convert to MIDI
    midi_path = notes_to_midi(generated_notes, bpm=bpm)
    
    # Convert to audio
    audio_path = midi_to_audio(midi_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print(f"\n📁 Output Files:")
    print(f"   MIDI: {midi_path}")
    if audio_path:
        print(f"   Audio: {audio_path}")
    
    print(f"\n🎉 Next Steps:")
    print(f"   1. Open MIDI in GarageBand/MuseScore to hear it")
    if audio_path:
        print(f"   2. Or play the audio file directly")
    print(f"   3. Run `streamlit run app.py` for the full UI")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
