"""Second test set: 220 new WikiArt paintings (44 per style), used once.

Why: on the first test, the fine-tuned ResNet18 beat the frozen one we had chosen on validation. To see
whether this was luck, we draw 220 paintings that no model and no notebook has ever seen.

Rules, in the spirit of our first split:
- none of our 1 500 paintings, and only WikiArt shards that we never opened;
- no painter known from our 1 500 paintings, so the models never saw these painters (most are
  "Unknown Artist": WikiArt names only 129 painters);
- at most 8 paintings per style from one shard (spread over WikiArt) and 10 per named painter;
- the same processing as make_wikiart_subset.py (224 x 224 centre crop, JPEG quality 90);
- then a near-duplicate check (perceptual hash) against our 1 500 paintings: a close copy is removed.

Run from the repository root:  python scripts/make_second_test.py [--dry-run]
The images go to data/wikiart_test2 (ignored by Git, like data/wikiart_style5).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_wikiart_subset as mws  # noqa: E402

CLASSES = ["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"]
FIRST = ROOT / "data" / "wikiart_style5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--per-class", type=int, default=44, help="paintings per style")
    p.add_argument("--seed", type=int, default=11, help="seed of the draw (different from the first subset)")
    p.add_argument("--per-shard", type=int, default=8, help="at most this many paintings of one style per shard")
    p.add_argument("--max-per-artist", type=int, default=10, help="at most this many paintings per named painter")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "wikiart_test2")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="print the plan without downloading")
    return p.parse_args()


def make_plan(index, class_ids, args):
    """Choose the rows: {shard: [rows]}, never from our 1 500 paintings, their shards or their painters."""
    names, shards = index["names"], index["shards"]
    unknown = names["artist"].index(mws.UNKNOWN_ARTIST)
    with open(FIRST / "metadata.csv", encoding="utf-8") as f:
        first = list(csv.DictReader(f))
    used_shards = {int(m["shard"]) for m in first}
    known = {names["artist"].index(m["artist"]) for m in first if m["artist"] != mws.UNKNOWN_ARTIST}
    rng = random.Random(args.seed)
    order = [s for s in range(len(shards)) if s not in used_shards]
    rng.shuffle(order)
    need, per_artist, plan = {c: args.per_class for c in class_ids}, Counter(), {}
    for s in order:
        rows = [r for r, (st, ar) in enumerate(zip(shards[s]["style"], shards[s]["artist"])) if st in need and ar not in known]
        rng.shuffle(rows)
        here, take = Counter(), []
        for r in rows:
            c, a = shards[s]["style"][r], shards[s]["artist"][r]
            if need[c] == 0 or here[c] >= args.per_shard:
                continue
            if a != unknown and per_artist[a] >= args.max_per_artist:
                continue
            take.append(r)
            need[c] -= 1
            here[c] += 1
            per_artist[a] += 1
        if take:
            plan[s] = sorted(take)
        if not any(need.values()):
            break
    return plan, need, len(known), len(used_shards)


def dhash(path):
    """64-bit perceptual hash: is each pixel brighter than its right neighbour, on a 9 x 8 grey image."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def near_duplicates(out, max_bits=6):
    """Images of the new set whose hash is within max_bits of one of our 1 500 paintings."""
    first = [FIRST / p for p in sorted(str(q.relative_to(FIRST)) for q in FIRST.rglob("*.jpg"))]
    h1 = np.stack([dhash(p) for p in first])
    with open(out / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    found = []
    for row in rows:
        bits = np.unpackbits(np.bitwise_xor(h1, dhash(out / row["path"])), axis=1).sum(axis=1)
        k = int(bits.argmin())
        if bits[k] <= max_bits:
            found.append({"id": row["id"], "closest": first[k].stem, "bits": int(bits[k])})
    return rows, found


def main():
    args = parse_args()
    index = mws.load_label_index(mws.REVISION, mws.default_index_path(mws.REVISION), args.retries)
    names = index["names"]["style"]
    class_ids = mws.resolve_classes(CLASSES, names)
    class_names = {c: names[c] for c in class_ids}
    plan, need, n_known, n_used = make_plan(index, class_ids, args)
    counts = Counter(class_names[index["shards"][s]["style"][r]] for s, rows in plan.items() for r in rows)
    print(f"Excluded: our 1 500 paintings, {n_used} shards already opened, {n_known} known painters.")
    print(f"Plan: {sum(counts.values())} paintings from {len(plan)} shards {sorted(plan)}, about "
          f"{len(plan) * mws.GB_PER_SHARD:.1f} GB to stream | {dict(sorted(counts.items()))}")
    if any(need.values()):
        print("Warning: missing", need)
    if args.dry_run:
        return
    for c in class_ids:
        (args.out / class_names[c]).mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(out=args.out, size=224, no_crop=False, quality=90, retries=args.retries, revision=mws.REVISION,
                            per_class=args.per_class, seed=args.seed, max_per_artist=args.max_per_artist,
                            strategy=f"second test: unused shards, at most {args.per_shard} per style per shard, no known painter")
    sizes = mws.download(plan, index, "style", class_names, ns, mws.previous_sizes(args.out))
    mws.write_metadata(plan, index, "style", class_ids, class_names, ns, sizes)
    rows, found = near_duplicates(args.out)
    if found:  # a close copy of a painting we trained on would make the test too easy: remove it
        drop = {d["id"] for d in found}
        with open(args.out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows([r for r in rows if r["id"] not in drop])
    info_path = args.out / "subset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["near_duplicates_removed"] = found
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Near-duplicates of our 1 500 paintings removed: {len(found)} {found if found else ''}")
    print(f"Second test set: {len(rows) - len(found)} paintings in {args.out}")


if __name__ == "__main__":
    main()
