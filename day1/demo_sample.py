import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

data_set =[
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

data_set = [s + " <END>" for s in data_set]
text =" ".join(data_set)
#print(text)

words = sorted(set(text.split()))
print(words)

vocab_size= len(words) #72

print(vocab_size)
words2index ={ w: i  for i, w  in enumerate(words)}
print(words2index)
idx2words ={ i: w  for w, i  in words2index.items()}
#print(ids2words)

data = torch.tensor([words2index[w] for w in text.split()],dtype=torch.long)
print(data)
print(len(data))#97

''' Data:

tensor([17, 35,  9, 23,  5, 41, 68, 33, 26, 70, 71, 49, 25, 34, 19, 33, 42, 37,
        11, 59, 43, 65, 27, 18,  3, 58, 33, 67, 70, 55, 24, 40, 28, 29, 59, 63,
        33, 57,  2, 66, 39,  8, 15, 59,  4,  6, 33, 44, 45, 60,  0, 16, 36, 50,
        20, 10, 33, 51,  8, 28, 29,  1, 56, 13, 46, 59, 53, 33, 53, 70, 64, 52,
        61, 21, 31, 12, 33, 62, 29, 52, 47, 30, 21, 31, 22, 14, 70, 38, 33, 69,
        47, 54, 31, 32,  7, 48, 33])

engineer - 0:  [32 values ]
have - 2: 32 dimensional vector array [0.64,0.16, 0.81...... ]


'''

block_size=6  #context lenght
embedding_dim=32  # evry words will have embeded vector holsing 32 
n_heads=2
n_layers=2
lr= 1e-3
epochs=1500

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y




class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generator(self, idx, max_new_tokens):
        end_token=words2index["<END>"]
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)

            if next_idx.item() == end_token:
                break

            idx = torch.cat((idx, next_idx), dim=1)
        return idx



mymodel = NanoGPT()
optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = mymodel(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")



context = torch.tensor([[words2index["Delhi"]]], dtype=torch.long)
out_data = mymodel.generator(context, max_new_tokens=10)

print("\nGenerated text:\n")
print(" ".join(idx2words[int(i)] for i in out_data[0]))