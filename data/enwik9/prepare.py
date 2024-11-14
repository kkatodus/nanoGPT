import os
import pickle
import numpy as np
import zipfile
import argparse
import resource  # Works on Unix-based systems

# parsing arguments to decide which part of the dataset to output
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=str, default='train')
args = parser.parse_args()

# loading the enwik9 dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik9.zip')
data = zipfile.ZipFile(input_file_path).read('enwik9')
data = data.decode('utf-8')

print('Length of enwik9: {}'.format(len(data)))
# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define chunk size (e.g., process in chunks of 10 MB)
chunk_size = 10 * 1024 * 1024  # 10 MB

print("Outputting the {} part of the dataset".format(args.part))
if args.part == 'train':
    data = data[:int(len(data) * 0.9)]
elif args.part == 'val':
    data = data[int(len(data) * 0.9):int(len(data) * 0.95)]
elif args.part == 'test':
    data = data[int(len(data) * 0.95):]
else:
    raise ValueError("Unknown part of the dataset, use 'train', 'val' or 'test'")

n = len(data)
print(f"Total {args.part} data size: {n:,} characters")

# Open the binary file to write encoded data in chunks
output_file_path = os.path.join(os.path.dirname(__file__), f'{args.part}.bin')
with open(output_file_path, 'wb') as f:
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        encoded_chunk = np.array([stoi[c] for c in chunk], dtype=np.uint16)
        encoded_chunk.tofile(f)
    print(f"Saved encoded {args.part} data to {output_file_path}")

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
