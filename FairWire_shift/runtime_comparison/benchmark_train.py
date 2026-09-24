"""One-epoch timing probe; never represents training to convergence."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2] / "FairWire_fairness_loss"
sys.path.insert(0, str(ROOT))
import torch
import yaml
import dgl
from torch.utils.data import DataLoader
from data import load_dataset, preprocess
from Model import ModelSync
from setup_utils import set_seed


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def benchmark(dataset):
    set_seed(0)
    device = torch.device('cuda:0')
    cfg_path = ROOT / 'configs' / dataset / 'train_Sync.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg_train = cfg['train']
    alpha_a = {'cora': 10.0, 'citeseer': 0.1, 'amazon_photo': 0.05}[dataset]
    started = sync_time()
    g = load_dataset(dataset)
    values = preprocess(g)
    X, s, y, E, xm, sm, ym, em, xcs, xcy, ycs, pvals = values
    X, s, E, xm, sm, em = [v.to(device) for v in (X, s, E, xm, sm, em)]
    N = g.num_nodes()
    dst, src = torch.triu_indices(N, N, offset=1, device=device)
    edges = torch.stack([dst, src], dim=1)
    train_loader = DataLoader(edges.cpu(), batch_size=cfg_train['batch_size'],
                              num_workers=4, shuffle=True)
    val_loader = DataLoader(edges, batch_size=cfg_train['val_batch_size'], shuffle=False)
    model = ModelSync(X_marginal=xm, s_marginal=sm, y_marginal=ym,
                      E_marginal=em, num_nodes=N, p_values=pvals,
                      y_cond_s_marginal=ycs, gnn_X_config=cfg['gnn_X'],
                      gnn_E_config=cfg['gnn_E'], **cfg['diffusion']).to(device)
    ox = torch.optim.AdamW(model.graph_encoder.pred_X.parameters(), **cfg['optimizer_X'])
    oe = torch.optim.AdamW(model.graph_encoder.pred_E.parameters(), **cfg['optimizer_E'])
    preparation_s = sync_time() - started

    def update(batch):
        batch = batch.to(device)
        bd, bs = batch.T
        lx, fx, le, fe = model.log_p_t(X, E, bs, bd, E[bd, bs], s, y)
        loss = lx + le + alpha_a * fe
        ox.zero_grad()
        oe.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.graph_encoder.pred_X.parameters(), cfg_train['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(model.graph_encoder.pred_E.parameters(), cfg_train['max_grad_norm'])
        ox.step()
        oe.step()
        # Retain train.py's loss .item() synchronization, without external logging.
        return lx.item(), le.item()

    model.train()
    update(edges[:cfg_train['batch_size']])
    sync_time()
    torch.cuda.reset_peak_memory_stats()
    started = sync_time()
    progress = started
    for i, batch in enumerate(train_loader):
        update(batch)
        if time.perf_counter() - progress > 20:
            print(json.dumps({'dataset': dataset, 'phase': 'training', 'batch': i+1,
                              'total_batches': len(train_loader)}), flush=True)
            progress = time.perf_counter()
    train_s = sync_time() - started
    print(json.dumps({'dataset': dataset, 'training_epoch_seconds': train_s}), flush=True)
    model.eval()
    started = sync_time()
    progress = started
    for i, batch in enumerate(val_loader):
        bd, bs = batch.T
        result = model.val_step(X, E, s, y, bs, bd, E[bd, bs], is_diff_X=True)
        result[3].cpu().detach().numpy()
        result[5].cpu().detach().numpy()
        if time.perf_counter() - progress > 20:
            print(json.dumps({'dataset': dataset, 'phase': 'validation', 'batch': i+1,
                              'total_batches': len(val_loader)}), flush=True)
            progress = time.perf_counter()
    val_s = sync_time() - started
    row = {'dataset': dataset, 'nodes': N, 'features': X.shape[0],
           'candidate_pairs': len(edges), 'training_batch_size': cfg_train['batch_size'],
           'validation_batch_size': cfg_train['val_batch_size'],
           'training_batches': len(train_loader), 'validation_batches': len(val_loader),
           'alpha_A': alpha_a, 'alpha_X': 0.0, 'T': cfg['diffusion']['T'],
           'validation_every_epochs': cfg_train['val_every_epochs'],
           'preparation_seconds_excluded': preparation_s, 'warmup_training_batches_excluded': 1,
           'training_epoch_seconds': train_s, 'validation_pass_seconds': val_s,
           'amortized_epoch_seconds': train_s + val_s / cfg_train['val_every_epochs'],
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
           'full_training_seconds': None,
           'measurement': 'one early training epoch plus one validation pass; not convergence'}
    print(json.dumps(row), flush=True)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'amazon_photo'])
    parser.add_argument('--output', default=str(Path(__file__).with_name('training_results.json')))
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = {'gpu': torch.cuda.get_device_name(0),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'torch': torch.__version__, 'dgl': dgl.__version__, 'cpu_threads': 4,
              'seed': 0, 'repetitions': 1,
              'excluded': ['dataset/model preparation', 'one warmup update', 'W&B logging',
                           'checkpoint save', 'scheduler update'], 'results': []}
    for name in args.datasets:
        output['results'].append(benchmark(name))
        Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
        gc.collect()
        torch.cuda.empty_cache()
