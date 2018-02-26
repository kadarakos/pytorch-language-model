import argparse
import time
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
from lm import repackage_hidden, LM_LSTM
import reader
import numpy as np
import sys
from pprint import pprint
from vis import BoundaryVisualizer

seed = 101010

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Simplest LSTM-based language model in PyTorch')
parser.add_argument('--data_set', type=str, default='ptb', choices=['ptb', 'text8', 'coco'])
parser.add_argument('--data_path', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--embedding_size', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--num_steps', type=int, default=35,
                    help='number of LSTM steps')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability')
parser.add_argument('--init', type=str, choices=["orthogonal", "pytorch"], default="pytorch",
                    help='Learning rule')
parser.add_argument('--optimizer', type=str, choices=["sgd", "adam"], default="sgd",
                    help='Learning rule')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', type=str, choices=["default", "none"], default="default",
                    help='Type of learning rate schedule to use.')
parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='Gradient clipping bound.')
parser.add_argument('--patience', type=int,  default=2,
                    help='Number of epochs to continue running without improvement on the validation set')
parser.add_argument('--save', type=str,  default='lm_model.pt',
                    help='path to save the final model')
parser.add_argument('--checkpoint', type=str,
                    help='path to model checkpoint')
parser.add_argument('--eval', action='store_true',
                    help='only run evaluation')
parser.add_argument('--vis_bounds', action='store_true',
                    help='Run visualization ')
args = parser.parse_args()
pprint(args)

criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, epoch, lr):
  """Sets learning rate for optimizer."""
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def run_epoch(model, data, optimizer, is_train=False):
  """Runs the model on the given data."""
  if is_train:
    model.train()
  else:
    model.eval()
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  hidden = model.init_hidden()
  costs = 0.0
  iters = 0
  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
    optimizer.zero_grad()
    hidden = repackage_hidden(hidden)
    outputs, hidden = model(inputs, hidden)
    targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
    tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

    loss = criterion(outputs.view(-1, model.vocab_size), tt)
    costs += loss.data[0] * model.num_steps
    iters += model.num_steps
    # Report perplexity for PTB or BPC for Text8
    metric = "perplexity" if args.data_set == "ptb" else "bpc"
    perf = np.exp(costs / iters) if args.data_set == "ptb" else 1.4427 * (costs / iters)
    if is_train:
      loss.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      optimizer.step()
      if step % (epoch_size // 10) == 10:
        if args.vis_bounds:
            visualizer.run_vis(1)  
        print("{} : {} {:8.4f} speed: {} wps".format(step * 1.0 / epoch_size, metric, perf,
                                                       iters * model.batch_size / (time.time() - start_time)))
  return perf


if __name__ == "__main__":
  if args.data_set == 'text8':
    raw_data = reader.text8_raw_data(data_path=args.data_path)
  elif args.data_set == 'coco':
      raw_data = reader.coco_raw_data(data_path=args.data_path)
  else:
    raw_data = reader.ptb_raw_data(data_path=args.data_path)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
  vocab_size = len(word_to_id)
  print('Vocabluary size: {}'.format(vocab_size))
  if args.eval:
    model = torch.load(args.checkpoint)
    model.batch_size = args.batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr)
    print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data, optimizer)))
    sys.exit()
  lr = args.initial_lr
  # decay factor for learning rate
  lr_decay_base = 1 / 1.15
  # we will not touch lr for the first m_flat_lr epochs
  m_flat_lr = 14.0
  model = LM_LSTM(embedding_dim=args.embedding_size, hidden_size=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                  vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)
  
  if args.vis_bounds:
    p = args.checkpoint if args.eval else args.save
    visualizer = BoundaryVisualizer(args.data_set,  model, raw_data)
    global visualizer
  model.cuda()
  print(model)
  print("########## Training ##########################")
  if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
  notimprove = 0
  best_val = run_epoch(model, valid_data, optimizer)
  for epoch in range(args.num_epochs):
    is_best = False
    if args.lr_schedule == "default":
      lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
      lr = lr * lr_decay # decay lr if it is time
      adjust_learning_rate(optimizer, epoch, lr)
    train_p = run_epoch(model, train_data, optimizer, True)
    val_ppl = run_epoch(model, valid_data, optimizer)
    print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
    is_best = val_ppl < best_val
    if is_best:
      print("Saving checkpoint to {}".format(args.save))
      best_val = val_ppl
      notimprove = 0
      with open(args.save, 'wb') as f:
        torch.save(model, f)
    else:
        notimprove += 1
    if notimprove == args.patience:
      print("Breaking, haven't seen improvement in {} epochs".format(notimprove))
      break
    print('Validation perplexity at epoch {}: {:8.2f}, best: {:8.2f}'.format(epoch, val_ppl, best_val))
  print("########## Testing ##########################")
  model.batch_size = 1 # to make sure we process all the data
  print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data, optimizer)))
  print("########## Done! ##########################")
