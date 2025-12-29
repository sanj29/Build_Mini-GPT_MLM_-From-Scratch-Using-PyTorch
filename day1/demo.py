import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

data_set =[
    "Hi how are you, hope doing greate",
    "This is mini language model basic learnung",
    "It's festive seasion in New York city, Happy new year",
    "Delhi is had foggy weather, see you in delhi",
    "Let have tea, love to take in winter morning",
    "I am QA engineer, learing LLM building from sractch",
    "Love to see you all soon, let meet in Bangalore",
    "Bangalore is also know as city of garden",
    "Do you know the pink city of India, it is Jaipur",
    "That the end of data-set for now"

]

data_set = [s + " <END>" for s in data_set]
text =" ".join(data_set)
#print(text)

words = list(set(text.split()))
print(words)

vocab_size= len(words) #72

print(vocab_size)
words2ndex ={ w: i  for i, w  in enumerate(words)}
print(words2ndex)
ids2words ={ i: w  for w, i  in words2ndex.items()}
#print(ids2words)

data = torch.tensor([words2ndex[w] for w in text.split()],dtype=torch.long)
print(data)
print(len(data))#97

''' Data:

ensor([17, 35,  9, 23,  5, 41, 68, 33, 26, 70, 71, 49, 25, 34, 19, 33, 42, 37,
        11, 59, 43, 65, 27, 18,  3, 58, 33, 67, 70, 55, 24, 40, 28, 29, 59, 63,
        33, 57,  2, 66, 39,  8, 15, 59,  4,  6, 33, 44, 45, 60,  0, 16, 36, 50,
        20, 10, 33, 51,  8, 28, 29,  1, 56, 13, 46, 59, 53, 33, 53, 70, 64, 52,
        61, 21, 31, 12, 33, 62, 29, 52, 47, 30, 21, 31, 22, 14, 70, 38, 33, 69,
        47, 54, 31, 32,  7, 48, 33])

engineer - 0:  [32 values ]
have - 2: 32 dimensional vector array [0.64,0.16, 0.81...... ]


'''

block_size=6  #conytext lenght
embeddung_dimi=32  # evry words will have embeded vector holsing 32 
n_heads=2
n_layers=2
ir=-1e-2
epochs=1500

def get_batch(batch_size=16):
    ix=torch.randInt(len(data)-block_size,(block_size,)) # size of data: 97, and block_size=6, so 97-6= 91 ( 0-90)
    x= torch.stack([data[i:i+block_size]] for i in ix)
    y= torch.stack([data[i:i+block_size+1]] for i in ix)

    return x,y