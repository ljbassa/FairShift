import os
import pickle as pkl
import random
import argparse
import networkx as nx
from pathlib import Path
import hashlib
import numpy as np

def add_tag_before_ext(path_str: str, tag: str) -> str:
    p = Path(path_str)
    return str(p.with_name(p.stem + tag + p.suffix))

def ppr_sweep_sample_nodes(g: nx.Graph, k: int, seed: int, restart_prob: float = 0.15, oversample: float = 3.0):
    rng = random.Random(seed)
    start = rng.choice(list(g.nodes()))

    personalization = {n: 0.0 for n in g.nodes()}
    personalization[start] = 1.0

    pr = nx.pagerank(g, alpha=1.0 - restart_prob, personalization=personalization)
    order = sorted(g.nodes(), key=lambda n: pr[n] / (g.degree(n) + 1e-9), reverse=True)

    total_vol = sum(dict(g.degree()).values())
    S = set()
    vol = 0
    cut = 0
    best = None 

    max_size = int(min(len(order), max(k, int(k * oversample))))
    for i in range(max_size):
        u = order[i]
        for v in g.neighbors(u):
            if v in S:
                cut -= 1
            else:
                cut += 1
        S.add(u)
        vol += g.degree(u)

        size = i + 1
        if size < max(20, k // 5):
            continue

        denom = min(vol, total_vol - vol)
        if denom <= 0:
            continue
        phi = cut / denom
        score = phi + 0.02 * abs(size - k) / k

        if best is None or score < best[0]:
            best = (score, size)

    chosen = order[: best[1]] if best else order[:k]

    if len(chosen) > k:
        chosen = chosen[:k]
    elif len(chosen) < k:
        need = k - len(chosen)
        rest = [n for n in order if n not in set(chosen)]
        chosen = chosen + rest[:need]

    return chosen

def multi_bfs_sample_nodes(g: nx.Graph, k: int, seed: int, num_seeds: int = 3) -> list:
    rng = random.Random(seed)
    nodes = list(g.nodes())

    starts = rng.sample(nodes, num_seeds) if len(nodes) >= num_seeds else rng.choices(nodes, k=num_seeds)

    selected = []
    selected_set = set()

    queues = [[s] for s in starts]
    active = list(range(len(queues)))

    while len(selected) < k and active:
        next_active = []
        for idx in active:
            if len(selected) >= k:
                break
            q = queues[idx]
            found = False
            while q:
                u = q.pop(0)
                if u in selected_set:
                    continue
                selected_set.add(u)
                selected.append(u)
                found = True

                nbrs = list(g.neighbors(u))
                rng.shuffle(nbrs)
                for v in nbrs:
                    if v not in selected_set:
                        q.append(v)
                break

            if q or found:
                next_active.append(idx)
        active = next_active

    if len(selected) < k:
        remaining = [n for n in nodes if n not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: (k - len(selected))])

    return selected

def bfs_sample_nodes(g: nx.Graph, k: int, seed: int) -> list:
    rng = random.Random(seed)
    nodes = list(g.nodes())
    start = rng.choice(nodes)

    selected = []
    visited = set()
    queue = [start]

    while queue and len(selected) < k:
        u = queue.pop(0)
        if u in visited:
            continue
        visited.add(u)
        selected.append(u)

        nbrs = list(g.neighbors(u))
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in visited:
                queue.append(v)

    if len(selected) < k:
        remaining = [n for n in nodes if n not in visited]
        rng.shuffle(remaining)
        selected.extend(remaining[: (k - len(selected))])

    return selected

def random_sample_nodes(g: nx.Graph, k: int, seed: int) -> list:
    rng = random.Random(seed)
    nodes = list(g.nodes())
    rng.shuffle(nodes)
    return nodes[:k]


def induced_relabel(g: nx.Graph, chosen: list) -> nx.Graph:
    sub = g.subgraph(chosen).copy()

    mapping = {old: i for i, old in enumerate(chosen)}
    sub = nx.relabel_nodes(sub, mapping, copy=True)
    nx.set_node_attributes(sub, {mapping[old]: old for old in chosen}, "orig_id")
    return sub

def drop_x_attr(g: nx.Graph) -> nx.Graph:
    g2 = g.copy()
    for n in g2.nodes():
        if "x" in g2.nodes[n]:
            del g2.nodes[n]["x"]
    return g2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.1, help="fraction of nodes to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", type=str, default="bfs", choices=["bfs", "bfs_n", "ppr", "random"])
    ap.add_argument("--num_seeds", type=int, default=3, help="number of BFS starting points (only for bfs_n)")
    ap.add_argument("--in_feat", type=str, default="graphs/cora_feat.pkl")
    ap.add_argument("--in_plain", type=str, default="graphs/cora.pkl")
    # out_feat/plain은 이제 ratio에 따라 자동 생성되므로 기본값은 디렉토리 참조용으로만 둡니다.
    ap.add_argument("--out_dir", type=str, default="graphs", help="output directory")
    args = ap.parse_args()

    # 1) Load Graph
    if os.path.exists(args.in_feat):
        with open(args.in_feat, "rb") as f:
            g_src = pkl.load(f)
        src_kind = "feat"
    else:
        with open(args.in_plain, "rb") as f:
            g_src = pkl.load(f)
        src_kind = "plain(no x)"

    n = g_src.number_of_nodes()
    k = max(2, int(round(n * args.ratio)))

    # 2) Sample Nodes
    if args.method == "ppr":
        chosen = ppr_sweep_sample_nodes(g_src, k=k, seed=args.seed)
    elif args.method == "bfs":
        chosen = bfs_sample_nodes(g_src, k=k, seed=args.seed)
    elif args.method == "bfs_n":
        chosen = multi_bfs_sample_nodes(g_src, k=k, seed=args.seed, num_seeds=args.num_seeds)
    else:
        chosen = random_sample_nodes(g_src, k=k, seed=args.seed)

    # 3) Induced Subgraph
    g_sub = induced_relabel(g_src, chosen)

    # Hash Check (Optional Debugging)
    def hash_x(x):
        arr = np.asarray(list(x), dtype=np.float32)
        return hashlib.md5(arr.tobytes()).hexdigest()

    if src_kind == "feat":
        total = 0
        bad = 0
        for new_id in g_sub.nodes():
            orig = g_sub.nodes[new_id].get("orig_id", None)
            if orig is None: continue
            x_new = g_sub.nodes[new_id].get("x", None)
            x_old = g_src.nodes[orig].get("x", None)
            if x_new is None or x_old is None:
                bad += 1; total += 1; continue

            total += 1
            if hash_x(x_new) != hash_x(x_old):
                bad += 1
        if bad > 0:
            print(f"[Debug] Hash check: total={total}, bad={bad}")

    # 4) 저장 파일명 자동 생성 (ratio 반영)
    # Ratio 문자열: 0.1 -> "01", 0.2 -> "02", 0.05 -> "005"
    ratio_str = f"{args.ratio:g}".replace('.', '')
    
    # 입력 파일명("cora") 추출
    p_in = Path(args.in_feat)
    base_name = p_in.stem  # "cora_feat"
    if base_name.endswith("_feat"):
        base_name = base_name[:-5] # "cora"

    # 기본 이름: cora_02
    base_with_ratio = f"{base_name}_{ratio_str}"

    # Method Tag 설정
    tag = args.method
    if args.method == "bfs_n":
        tag = f"bfs_multi{args.num_seeds}"
    
    # 출력 경로 조립
    out_dir_path = Path(args.out_dir)
    os.makedirs(out_dir_path, exist_ok=True)

    # 예: graphs/cora_02_bfs_multi3_feat.pkl
    out_feat = str(out_dir_path / f"{base_with_ratio}_{tag}_feat{p_in.suffix}")
    # 예: graphs/cora_02_bfs_multi3.pkl
    out_plain = str(out_dir_path / f"{base_with_ratio}_{tag}{p_in.suffix}")

    with open(out_feat, "wb") as f:
        pkl.dump(g_sub, f, protocol=pkl.HIGHEST_PROTOCOL)

    g_plain = drop_x_attr(g_sub)
    with open(out_plain, "wb") as f:
        pkl.dump(g_plain, f, protocol=pkl.HIGHEST_PROTOCOL)

    # 5) Summary
    has_x = any(("x" in g_sub.nodes[i]) for i in g_sub.nodes())
    xdim = None
    if has_x:
        for i in g_sub.nodes():
            if "x" in g_sub.nodes[i]:
                xdim = len(g_sub.nodes[i]["x"])
                break

    print(f"[OK] source={src_kind}, N={n} -> k={k} (ratio={args.ratio}, method={args.method}, seed={args.seed})")
    print(f"     subgraph: nodes={g_sub.number_of_nodes()}, edges={g_sub.number_of_edges()}")
    print(f"     saved: {out_feat}")
    print(f"     saved: {out_plain}")

if __name__ == "__main__":
    main()