import os
import io
import argparse
import math
import lmdb
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm

def list_keys(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=4096)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor()]
    env.close()
    return keys

def worker(args):
    lmdb_path, keys, out_dir, start_idx, fmt, quality, skip_existing = args
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=4096)
    cnt = 0
    with env.begin() as txn:
        for i, k in enumerate(keys):
            outname = os.path.join(out_dir, f"{start_idx + i:08d}.{fmt}")
            if skip_existing and os.path.exists(outname):
                cnt += 1
                continue
            v = txn.get(k)
            if not v:
                continue
            try:
                img = Image.open(io.BytesIO(v))
                img.save(outname) if fmt=="png" else img.convert("RGB").save(outname, "JPEG", quality=95, subsampling=0)
                cnt += 1
            except Exception:
                continue
    env.close()
    return cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="jpg", choices=["jpg","png"])
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keys = list_keys(args.lmdb)
    n = len(keys)

    W = max(1, args.workers)
    chunk = math.ceil(n / W)
    tasks = []
    for w in range(W):
        s, e = w*chunk, min(n, (w+1)*chunk)
        if s >= e: break
        tasks.append((args.lmdb, keys[s:e], args.out, args.start_index + s, args.fmt, args.quality, args.skip_existing))

    with mp.Pool(processes=W) as pool:
        pbar = tqdm(total=n)
        done = 0
        for c in pool.imap_unordered(worker, tasks):
            done += c; pbar.update(c)
        pbar.close()
    print(f"[INFO] Exported {done}/{n} -> {args.out}")

if __name__ == "__main__":
    main()
