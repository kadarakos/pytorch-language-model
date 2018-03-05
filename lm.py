import torch.nn as nn
from torch.autograd import Variable
# from denura import topdown, ran, hmlstm, simple_ran
from denura import topdown, hmlstm, simple_ran
from denura.util import OneHot


class LM_LSTM(nn.Module):
  """Simple LSMT-based language model"""
  def __init__(self, embedding_dim, rnn_type, hidden_size, num_steps, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(LM_LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)

    if embedding_dim == 0:
        self.word_embeddings = OneHot(vocab_size)
        self.embedding_dim = vocab_size
    else:
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    if rnn_type == 'lstm':
        cell = nn.LSTM
    elif rnn.type == 'custom-lstm':
        cell = lstm.LSTM
    elif rnn.type == 'topdown':
        cell = topdown.TopDownLSTM
    elif rnn_type == 'hmlstm':
        cell = hmlstm.HMLSTM
    self.lstm = cell(input_size=self.embedding_dim,
                     hidden_size=self.hidden_size,
                     num_layers=num_layers,
                     dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=self.hidden_size,
                           out_features=vocab_size)
    self.init_weights()
  #TODO make orthogonal init optional
  def init_weights(self):
    init_range = 0.1
    if not isinstance(self.word_embeddings, OneHot):
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)

    self.sm_fc.bias.data.fill_(0.0)
    # self.sm_fc.weight.data.uniform_(-init_range, init_range)
    nn.init.orthogonal(self.sm_fc.weight)
    for name, param in self.lstm.named_parameters():
      if 'bias' in name:
        nn.init.constant(param, 0.0)
      elif 'weight' in name:
        nn.init.orthogonal(param)

  def init_hidden(self):
    weight = next(self.parameters()).data
    if isinstance(self.lstm, hmlstm.HMLSTM):
        Ht = [Variable(weight.new(self.batch_size, self.hidden_size).zero_()) for x in range(self.num_layers)]
        C = [Variable(weight.new(self.batch_size, self.hidden_size).zero_()) for x in range(self.num_layers)]
        Z = [Variable(weight.new(self.batch_size).zero_()) for x in range(self.num_layers - 1)]
        return [Ht, C, Z]
    else:
        return (Variable(weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()))

  def forward(self, inputs, hidden):
    embeds = self.dropout(self.word_embeddings(inputs))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)
    logits = self.sm_fc(lstm_out.view(-1, self.hidden_size))
    return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden

def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)
