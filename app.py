import streamlit as st
import torch
import numpy as np
import pickle
import os
from pathlib import Path
import music21
from music21 import instrument, note, chord, stream, tempo
import subprocess
import tempfile
from datetime import datetime

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="🎵 AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 AI Music Generator")
st.markdown("Generate unique melodies powered by your trained LSTM model")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "model.pth"
VOCAB_PATH = DATA_DIR / "vocab.pkl"
NOTE_TO_INT_PATH = DATA_DIR / "note_to_int.pkl"
INT_TO_NOTE_PATH = DATA_DIR / "int_to_note.pkl"
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_vocab():
    """Load the trained LSTM model and vocabulary"""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()

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
        st.error(
            "Vocabulary not found. Expected data/vocab.pkl or both "
            "data/note_to_int.pkl and data/int_to_note.pkl."
        )
        st.stop()

    vocab_size = len(note_to_int)
    
    # Load model (you'll need to define MusicLSTM here or import from your code)
    model = MusicLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    return model, note_to_int, int_to_note


class MusicLSTM(torch.nn.Module):
    """LSTM model for music generation"""
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


def generate_music(model, note_to_int, int_to_note, seed_note, num_notes=100, temperature=0.8):
    """Generate a sequence of notes from the model"""
    model.eval()
    
    sequence = [note_to_int[seed_note]]
    hidden = None
    
    with torch.no_grad():
        for _ in range(num_notes):
            input_tensor = torch.tensor([sequence[-1]], dtype=torch.long).unsqueeze(0)
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature
            logits = output[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Sample next note
            next_note_idx = np.random.choice(len(int_to_note), p=probabilities)
            sequence.append(next_note_idx)
    
    # Convert to note names
    generated_notes = [int_to_note[idx] for idx in sequence]
    return generated_notes


def notes_to_midi(note_sequence, bpm=120, output_path="generated_music.mid"):
    """Convert note sequence to MIDI file"""
    s = stream.Score()
    part = stream.Part()
    part.append(instrument.Piano())
    part.append(tempo.MetronomeMark(number=bpm))
    
    for note_name in note_sequence:
        try:
            if note_name == 'rest':
                n = note.Rest(quarterLength=0.5)
            else:
                n = note.Note(note_name, quarterLength=0.5)
            part.append(n)
        except:
            # Skip invalid notes
            continue
    
    s.append(part)
    s.write('midi', fp=output_path)
    return output_path


def find_soundfont():
    """Find a FluidSynth-compatible SoundFont file."""
    for soundfont in DATA_DIR.glob("*.sf[23]"):
        return soundfont

    for soundfont in SOUNDFONT_CANDIDATES:
        if soundfont and soundfont.exists():
            return soundfont

    return None


def midi_to_audio(midi_path, audio_path="generated_music.wav", soundfont=None):
    """Convert MIDI to audio using FluidSynth"""
    try:
        if soundfont is None:
            soundfont = find_soundfont()

        if not soundfont or not Path(soundfont).exists():
            return None

        cmd = [
            'fluidsynth',
            '-ni',
            '-T', 'wav',
            '-F', audio_path,
            '-r', '44100',
            str(soundfont),
            midi_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    except FileNotFoundError:
        st.warning("FluidSynth was not found. MIDI download is still available.")
        return None
    except Exception as e:
        st.warning(f"Audio conversion failed: {e}. MIDI download is still available.")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

# Sidebar Configuration
st.sidebar.header("🎼 Generation Settings")

# Load model
try:
    model, note_to_int, int_to_note = load_model_and_vocab()
    available_notes = sorted(list(note_to_int.keys()))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User inputs
seed_note = st.sidebar.selectbox(
    "🎹 Seed Note",
    available_notes,
    help="Start melody with this note"
)

num_notes = st.sidebar.slider(
    "📏 Number of Notes",
    min_value=20,
    max_value=500,
    value=100,
    step=10,
    help="Length of generated sequence"
)

temperature = st.sidebar.slider(
    "🌡️ Temperature (Creativity)",
    min_value=0.1,
    max_value=2.0,
    value=0.8,
    step=0.1,
    help="Lower = more predictable, Higher = more creative"
)

bpm = st.sidebar.slider(
    "🎯 Tempo (BPM)",
    min_value=40,
    max_value=200,
    value=120,
    step=5
)

uploaded_soundfont = st.sidebar.file_uploader(
    "SoundFont for audio preview",
    type=["sf2", "sf3"],
    help="Optional. Upload a General MIDI SoundFont to enable WAV playback."
)

detected_soundfont = find_soundfont()
if detected_soundfont:
    st.sidebar.caption(f"Audio preview: {detected_soundfont.name}")
elif uploaded_soundfont is None:
    st.sidebar.caption("Audio preview needs a .sf2 or .sf3 SoundFont. MIDI works without it.")

# Generate Button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🎵 Generate Music", use_container_width=True, key="generate"):
        st.session_state.generated = True

# Display generated music
if 'generated' in st.session_state and st.session_state.generated:
    st.divider()
    
    with st.spinner("🎼 Generating music..."):
        generated_notes = generate_music(
            model, note_to_int, int_to_note,
            seed_note=seed_note,
            num_notes=num_notes,
            temperature=temperature
        )
    
    # Show generated sequence
    st.success("✅ Music generated!")
    
    with st.expander("📊 View Generated Sequence"):
        st.write(f"First 30 notes: {generated_notes[:30]}")
    
    # Create temp files
    temp_dir = tempfile.mkdtemp()
    midi_path = os.path.join(temp_dir, "generated_music.mid")
    audio_path = os.path.join(temp_dir, "generated_music.wav")
    soundfont_path = detected_soundfont

    if uploaded_soundfont is not None:
        soundfont_path = os.path.join(temp_dir, uploaded_soundfont.name)
        with open(soundfont_path, "wb") as f:
            f.write(uploaded_soundfont.getbuffer())
    
    # Convert to MIDI
    with st.spinner("🎹 Converting to MIDI..."):
        midi_path = notes_to_midi(generated_notes, bpm=bpm, output_path=midi_path)
    
    # Convert to Audio
    audio_exists = False
    if soundfont_path:
        with st.spinner("🔊 Converting to audio (this may take a moment)..."):
            result = midi_to_audio(midi_path, audio_path=audio_path, soundfont=soundfont_path)
            if result:
                audio_exists = True
    else:
        st.info("MIDI is ready. Upload or add a .sf2/.sf3 SoundFont to enable audio preview.")
    
    # Display downloads
    st.markdown("### 📥 Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with open(midi_path, 'rb') as f:
            st.download_button(
                label="⬇️ Download MIDI",
                data=f.read(),
                file_name=f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid",
                mime="audio/midi",
                use_container_width=True
            )
    
    if audio_exists:
        with col2:
            with open(audio_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download MP3",
                    data=f.read(),
                    file_name=f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
    
    # Display audio player
    if audio_exists:
        st.markdown("### 🎧 Listen Here")
        with open(audio_path, 'rb') as f:
            st.audio(f.read(), format='audio/wav')

# Info section
st.divider()