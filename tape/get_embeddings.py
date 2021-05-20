import numpy as np
import torch
import sys
from tape import ProteinBertModel, TAPETokenizer

def parse_fasta(filename,limit=-1):
  '''function to parse fasta'''
  header = []
  sequence = []
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      if len(header) == limit:
        break
      header.append(line[1:])
      sequence.append([])
    else:
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return np.array(header), np.array(sequence)

### Create BERT model
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')
model.eval()

### run model and save results
a, b = parse_fasta(sys.argv[1])
token_ids = torch.tensor([tokenizer.encode(b[0])])
sequence_output, pooled_output = model(token_ids)
sequence_output = sequence_output.detach().numpy()
np.save(sys.argv[2], sequence_output)
