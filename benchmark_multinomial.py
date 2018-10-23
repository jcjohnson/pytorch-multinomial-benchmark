import os, argparse, json, random, time
import torch
import numpy as np


def int_list(s):
  return [int(x) for x in s.split(',')]


parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int_list,
          default=[1, 2, 4, 8, 16, 32, 64, 128])
parser.add_argument('--C', type=int_list,
          default=[10, 100, 1000, 10000, 100000])
parser.add_argument('--S', type=int_list,
          default=[10, 100, 1000, 10000, 100000])
parser.add_argument('--with_replacement', type=int, default=1)
parser.add_argument('--num_trials', type=int, default=5)
parser.add_argument('--stats_json', default='multinomial_stats.json')



def main(args):
  replacement = args.with_replacement == 1
  all_results = []
  for n, N in enumerate(args.N):
    print('Running N = %d (value %d / %d)' % (N, n + 1, len(args.N)))
    for c, C in enumerate(args.C):
      print('  Running C = %d (value %d / %d)' % (C, c + 1, len(args.C)))
      for s, S in enumerate(args.S):
        print('    Running S = %d (value %d / %d)' % (S, s + 1, len(args.S)))
        cur_results = {
          'N': N, 'C': C, 'S': S,
          'torch_cpu': [], 'torch_gpu': [], 'numpy_cpu': [], 'numpy_gpu': []
        }
        for t in range(args.num_trials):
          times = run_trial(N, C, S, replacement)
          for key, time_ms in times.items():
            cur_results[key].append(time_ms)
        all_results.append(cur_results)

  with open(args.stats_json, 'w') as f:
    json.dump(all_results, f)


def timeit(f, *args, **kwargs):
  torch.cuda.synchronize()
  t0 = time.time()
  out = f(*args, **kwargs)
  torch.cuda.synchronize()
  t1 = time.time()
  time_ms = (t1 - t0) * 1000.0
  return time_ms


def numpy_multinomial(probs, num_samples, replacement=True):
  N, C, S = probs.shape[0], probs.shape[1], num_samples
  probs_np = probs.cpu().numpy()
  samples = []
  for i in range(N):
    cur_probs = probs_np[i]
    cur_samples = np.random.choice(C, size=S, replace=replacement, p=cur_probs)
    samples.append(cur_samples[None])
  samples = np.concatenate(samples, axis=0)
  samples = torch.tensor(samples).long().to(probs.device)
  return samples


def run_trial(N, C, S, replacement=True):
  probs_cpu = torch.rand(N, C).softmax(dim=1)
  probs_gpu = probs_cpu.cuda()

  # We want to test torch and numpy on both cpu and gpu; randomize the order
  # in which we call them to minimize any systematic effects of caching, etc
  kwargs = {'replacement': replacement}
  calls = [
      ['torch_cpu', torch.multinomial, (probs_cpu, S), kwargs],
      ['torch_gpu', torch.multinomial, (probs_gpu, S), kwargs],
      ['numpy_cpu', numpy_multinomial, (probs_cpu, S), kwargs],
      ['numpy_gpu', numpy_multinomial, (probs_gpu, S), kwargs],
  ]
  random.shuffle(calls)
  
  results = {}
  for key, f, args, kwargs in calls:
    results[key] = timeit(f, *args, **kwargs)
  return results


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

