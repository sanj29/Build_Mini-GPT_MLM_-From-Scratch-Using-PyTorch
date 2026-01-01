import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_layers import TransformerBlock


# ----------------------------
# Environment Info
# ----------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# ----------------------------
# Dataset
# ----------------------------
sentences = [
    "Hi, how are you, hope doing great",
    "This is a mini language model, basic learning",
    "It's festive season in New York City, Happy New Year.",
    "Delhi has foggy weather, see you in Delhi",
    "Let's have tea, love to take in a winter morning",
    "I am a QA engineer, learning LLM building from scratch",
    "Love to see you all soon, let's meet in Bangalore",
    "Bangalore is also known as the city of gardens.",
    "Do you know the pink city of India? It is Jaipur.",
    "That's the end of the data set for now."
]

sentences = [s + " <END>" for s in sentences]
text = " ".join(sentences)

# Vocabulary
tokens = sorted(set(text.split()))
vocab_size = len(tokens)

stoi = {w: i for i, w in enumerate(tokens)}
itos = {i: w for w, i in stoi.items()}

data = torch.tensor([stoi[w] for w in text.split()], dtype=torch.long)

print("Vocab size:", vocab_size)
print("Total tokens:", len(data))


# ----------------------------
# Hyperparameters
# ----------------------------
block_size = 6
embed_dim = 32
num_heads = 2
num_layers = 2
learning_rate = 1e-3
epochs = 1500
batch_size = 16


# ----------------------------
# Batch Generator
# ----------------------------
def get_batch():
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ----------------------------
# Mini GPT Model
# ----------------------------
class NanoGPT(nn.Module):
    """
    Minimal GPT-style language model for educational purposes.
    """

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.norm_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = token_emb + pos_emb
        x = self.transformer(x)
        x = self.norm_final(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        end_token = stoi["<END>"]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)

            if next_idx.item() == end_token:
                break

            idx = torch.cat([idx, next_idx], dim=1)

        return idx


# ----------------------------
# Training
# ----------------------------
model = NanoGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(epochs):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


# ----------------------------
# Text Generation
# ----------------------------
context = torch.tensor([[stoi["Hi,"]]], dtype=torch.long)
generated = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(itos[int(i)] for i in generated[0]))
