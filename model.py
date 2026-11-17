# =============================================================
#  PHASE 3 — LSTM Model for AI Music Generation
#  Run: python model.py
#  Make sure preprocess.py has been run first (data/ folder exists)
# =============================================================

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================
#  PART 1: Load the preprocessed data
# =============================================================

def load_data(data_dir='data'):
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    with open(os.path.join(data_dir, 'note_to_int.pkl'), 'rb') as f:
        note_to_int = pickle.load(f)

    with open(os.path.join(data_dir, 'int_to_note.pkl'), 'rb') as f:
        int_to_note = pickle.load(f)

    vocab_size = len(note_to_int)
    print(f"Loaded {len(X)} sequences, vocab size: {vocab_size}")
    return X, y, note_to_int, int_to_note, vocab_size


# =============================================================
#  PART 2: PyTorch Dataset
#  PyTorch needs data wrapped in a Dataset object.
#  Think of it as a box that hands out (input, label) pairs.
# =============================================================

class MusicDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.long)    # note IDs (integers)
        self.y = torch.tensor(y, dtype=torch.long)    # target note ID

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================
#  PART 3: The LSTM Model
#
#  Architecture:
#    Embedding → LSTM x3 → Dropout → Linear → output
#
#  What each layer does:
#  - Embedding : turns integer note IDs into dense vectors
#                (like word embeddings in NLP)
#                e.g. note ID 42 → [0.3, -0.1, 0.8, ...]
#
#  - LSTM x3   : 3 stacked LSTM layers that learn patterns
#                over sequences. Each layer sees what the
#                previous layer learned.
#
#  - Dropout   : randomly zeroes 30% of neurons during
#                training — prevents memorising, forces
#                generalisation
#
#  - Linear    : final layer maps to vocab_size outputs.
#                Each output = score for "is this the next note?"
# =============================================================

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=3, dropout=0.3):
        super(MusicLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding: integer → dense vector
        # vocab_size possible notes, each mapped to embed_dim numbers
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM: processes the sequence of embeddings
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,       # dropout between LSTM layers
            batch_first=True
        )

        # Dropout before the final layer
        self.dropout = nn.Dropout(dropout)

        # Linear: hidden_dim → vocab_size (one score per possible note)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Step 1: embed note IDs into vectors
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)

        # Step 2: pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)

        # Step 3: take only the LAST timestep's output
        # (we only care about predicting the NEXT note, not all intermediate ones)
        last_output = lstm_out[:, -1, :]
        # last_output shape: (batch_size, hidden_dim)

        # Step 4: dropout + linear layer
        out = self.dropout(last_output)
        out = self.fc(out)
        # out shape: (batch_size, vocab_size)

        return out


# =============================================================
#  PART 4: Training loop
#
#  Each "epoch" = one full pass through all training data.
#  Steps per batch:
#    1. forward pass  → model makes predictions
#    2. compute loss  → how wrong were the predictions?
#    3. backward pass → compute gradients (blame each weight)
#    4. optimizer step → nudge weights in the right direction
# =============================================================

def train(model, dataloader, criterion, optimizer, device):
    model.train()           # put model in training mode
    total_loss = 0.0

    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Move data to GPU if available, else CPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward pass
        predictions = model(X_batch)
        # predictions shape: (batch_size, vocab_size)

        # 2. Compute loss
        # CrossEntropyLoss: measures how wrong the predictions are
        # It compares the predicted scores to the true next note
        loss = criterion(predictions, y_batch)

        # 3. Zero gradients (important! gradients accumulate otherwise)
        optimizer.zero_grad()

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Clip gradients — prevents "exploding gradient" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# =============================================================
#  PART 5: Generate music!
#
#  We give the model a "seed" sequence (50 notes),
#  then ask it to predict the next note, add it to the sequence,
#  and repeat for as many notes as we want.
#
#  Temperature controls creativity:
#    low  (0.5) = safe, repetitive, stays close to training data
#    high (1.2) = creative, unpredictable, sometimes chaotic
# =============================================================

def generate_music(model, seed_sequence, int_to_note, num_notes=100, temperature=1.0, device='cpu'):
    model.eval()   # switch to evaluation mode (disables dropout)

    generated = list(seed_sequence)  # start with the seed

    with torch.no_grad():  # no gradient tracking needed for generation
        for _ in range(num_notes):
            # Prepare input: last 50 notes as a tensor
            input_seq = torch.tensor([generated[-50:]], dtype=torch.long).to(device)

            # Get model's prediction (raw scores for each note)
            output = model(input_seq)

            # Apply temperature scaling
            # Divide by temperature before softmax:
            #   low temp  → sharper distribution (confident picks)
            #   high temp → flatter distribution (more random)
            output = output / temperature

            # Convert scores to probabilities
            probs = torch.softmax(output[0], dim=0)

            # Sample a note from the probability distribution
            next_note_id = torch.multinomial(probs, 1).item()

            generated.append(next_note_id)

    # Convert integer IDs back to note names
    generated_notes = [int_to_note[i] for i in generated[50:]]  # skip seed
    return generated_notes


# =============================================================
#  MAIN: Put it all together
# =============================================================

if __name__ == '__main__':

    # --- Config ---
    EPOCHS     = 50       # number of training passes (increase for better results)
    BATCH_SIZE = 64       # sequences processed at once
    LR         = 0.001    # learning rate (how big each weight update is)
    EMBED_DIM  = 64       # size of note embeddings
    HIDDEN_DIM = 256      # size of LSTM hidden state
    NUM_LAYERS = 3        # number of stacked LSTM layers

    # --- Device: use GPU if available, else CPU ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load data ---
    X, y, note_to_int, int_to_note, vocab_size = load_data()

    # --- Create dataset and dataloader ---
    dataset    = MusicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Build model ---
    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")

    # --- Loss function and optimiser ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training ---
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 40)

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, dataloader, criterion, optimizer, device)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  |  Loss: {loss:.4f}")

    print("-" * 40)
    print("Training complete!")

    # --- Save the trained model ---
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/model.pth')
    print("Model saved to data/model.pth")

    # --- Generate a sample ---
    print("\nGenerating 100 notes...")
    seed = list(X[0])   # use the first training sequence as seed
    notes = generate_music(model, seed, int_to_note, num_notes=100, temperature=1.0, device=device)

    print("\nGenerated note sequence (first 20 notes):")
    print(notes[:20])
    print("\nPhase 3 complete! Next → Phase 4: convert to MIDI and audio.")
