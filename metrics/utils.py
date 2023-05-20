"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) extract_time: Returns Maximum sequence length and each sequence length.
(3) Predictor: Basic RNN Cell for forecasting
"""

## Necessary Packages
import torch.nn as nn
import torch

def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - preprocessed_data: original preprocessed_data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len

class Predictor(nn.Module):
  def __init__(self, input_size, num_units, seq_length, pred_length):
    super(Predictor, self).__init__()

    self.gru = nn.GRU(input_size=input_size, hidden_size=num_units, batch_first=True)
    self.tanh = nn.Tanh()
    self.fc = nn.Linear(seq_length, pred_length)

  def forward(self, x, t):
    # Pass input through GRU cell
    gru_output1, gru_last_state1 = self.gru(x)
    gru_output1 = self.tanh(gru_output1)
    # Use pack_padded_sequence to handle variable-length sequences
    packed_output = nn.utils.rnn.pack_padded_sequence(gru_output1, t, batch_first=True, enforce_sorted=False)
    # Pass pack_padded_sequence through GRU cell
    gru_output2, gru_last_state2 = self.gru(packed_output.data.reshape(x.shape[0], x.shape[1], x.shape[2]))
    # Pass packed output through fully connected layer
    gru_output2 = gru_output2.permute((0, 2, 1))
    y_hat = self.fc(gru_output2)
    # shape : batch_size x features x pred_window
    # Pass through sigmoid activation
    y_hat_logit = torch.sigmoid(y_hat)

    return y_hat_logit.permute(0, 2, 1), y_hat