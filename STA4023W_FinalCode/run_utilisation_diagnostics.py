"""Run from Main. Uses existing unfiltered heuristics; never changes packing rules."""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import permutations


def volume(d):
    return math.prod(d)


def fits(d, b):
    return any(all(x <= y for x, y in zip(o, (b['length'], b['width'], b['height'])))
               for o in set(permutations(d)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyse_order(module, catalogue, order_id, items):
    bins, assigned = [], defaultdict(list)
    item_rows = []
    for pos, item in enumerate(items, 1):
        dims = tuple(item[k] for k in ('length', 'width', 'height'))
        feasible = [b for b in catalogue if fits(dims, b)]
        if not feasible:
            raise ValueError(f'{order_id}/{pos}: no feasible bin type')
        smallest = min(feasible, key=lambda b: volume((b['length'], b['width'], b['height'])))
        before = len(bins)
        b = module.pack_item(catalogue, bins, *dims)
        if b is None or not any(b is x for x in bins):
            raise ValueError(f'{order_id}/{pos}: pack_item failed')
        assigned[id(b)].append(dims)
        actual = tuple(b.placed_items[-1][a+3] - b.placed_items[-1][a] for a in range(3))
        if sorted(actual) != sorted(dims):
            raise ValueError(f'{order_id}/{pos}: placement dimensions differ from input')
        item_rows.append(dict(order_id=order_id, item_position=pos,
                              item_volume=volume(dims), opened_new_bin=int(len(bins)>before),
                              selected_bin_type=b.name,
                              smallest_feasible_bin_type=smallest['name'],
                              item_to_smallest_feasible_bin_pct=100*volume(dims)/volume((smallest['length'],smallest['width'],smallest['height']))))
    bin_rows = []
    for index, b in enumerate(bins, 1):
        placed = b.placed_items
        if len(placed) != len(assigned[id(b)]):
            raise ValueError(f'{order_id}: item accounting mismatch')
        used = 0
        for i, box in enumerate(placed):
            dims = tuple(box[a+3]-box[a] for a in range(3))
            if any(x <= 0 for x in dims):
                raise ValueError(f'{order_id}: nonpositive placement size')
            if any(box[a] < -1e-8 or box[a+3] > lim+1e-8
                   for a,lim in enumerate((b.length,b.width,b.height))):
                raise ValueError(f'{order_id}: placement outside bin')
            used += volume(dims)
            for other in placed[:i]:
                if all(box[a] < other[a+3]-1e-8 and other[a] < box[a+3]-1e-8 for a in range(3)):
                    raise ValueError(f'{order_id}: overlapping items')
        if not math.isclose(used, b.used_volume, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(f'{order_id}: cached volume mismatch')
        opening = volume(assigned[id(b)][0])
        bin_rows.append(dict(order_id=order_id, bin_index=index, bin_type=b.name,
                             items=len(placed), cost=b.cost, bin_volume=b.volume,
                             item_volume=used, unused_volume=b.volume-used,
                             utilisation_pct=100*used/b.volume,
                             opening_utilisation_pct=100*opening/b.volume,
                             opening_item_volume=opening))
    total = dict(order_id=order_id, items=len(items), bins=len(bins),
                 cost=sum(b.cost for b in bins), item_volume=sum(volume(d) for ds in assigned.values() for d in ds),
                 bin_volume=sum(b.volume for b in bins))
    if sum(r['items'] for r in bin_rows) != len(items):
        raise ValueError(f'{order_id}: total item count mismatch')
    total['utilisation_pct'] = 100*total['item_volume']/total['bin_volume']
    return total, bin_rows, item_rows


def summarise(rows, total_unused):
    capacity = sum(r['bin_volume'] for r in rows)
    used = sum(r['item_volume'] for r in rows)
    return dict(bins=len(rows), items=sum(r['items'] for r in rows),
                cost=sum(r['cost'] for r in rows), bin_volume=capacity, item_volume=used,
                unused_volume=capacity-used,
                weighted_utilisation_pct=100*used/capacity,
                mean_bin_utilisation_pct=sum(r['utilisation_pct'] for r in rows)/len(rows),
                single_item_bin_pct=100*sum(r['items']==1 for r in rows)/len(rows),
                weighted_opening_utilisation_pct=100*sum(r['opening_item_volume'] for r in rows)/capacity,
                share_of_all_unused_volume_pct=100*(capacity-used)/total_unused if total_unused else 0)


def main():
    from config import CONFIG
    from data_loader import load_orders
    from splits import ensure_split_manifest, select_orders
    from bin_types import bin_types
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=CONFIG.paths.raw_dataset)
    p.add_argument('--splits', type=Path, default=CONFIG.paths.split_manifest)
    p.add_argument('--output', type=Path, default=Path('results/utilisation_diagnostics'))
    p.add_argument('--limit', type=int, help='Smoke test only; first N test orders')
    p.add_argument('--heuristics', nargs='+', choices=['ha1','ha2','ha3'], default=['ha1','ha2','ha3'])
    p.add_argument('--reference-dir', type=Path, default=Path(__file__).resolve().parent/'utilisation_reference')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if not args.splits.is_file():
        p.error('Saved split manifest is required; no new split will be created.')
    if args.output.exists():
        p.error('Output already exists. Choose a new --output directory.')
    orders = load_orders(args.dataset)
    manifest = ensure_split_manifest(orders, args.splits)
    selected = select_orders(orders, manifest, 'test')
    if args.limit:
        selected = selected[:args.limit]
    args.output.mkdir(parents=True)
    metadata = dict(partition='test', orders=len(selected), partial_run=bool(args.limit),
                    bin_catalogue=bin_types, checks='orthogonal dimensions, containment, non-overlap, item count, cached volume, reference totals',
                    excluded_checks='weight, stability, blocking; not enforced by these baseline heuristics',
                    source_sha256={})
    summary_rows, grouped_rows, item_summary = [], [], []
    for name in args.heuristics:
        module = importlib.import_module('heuristics.'+name)
        metadata['source_sha256'][name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        ref_path = args.reference_dir/(name+'.csv')
        with ref_path.open(newline='', encoding='utf-8-sig') as f:
            ref = {r['order_id']:r for r in csv.DictReader(f)}
        totals, bins, pairs = [], [], []
        for n,(oid,items) in enumerate(selected,1):
            t,b,i = analyse_order(module, bin_types, oid, items)
            if oid not in ref:
                raise ValueError(f'{name}/{oid}: missing test reference')
            for k in ('items','bins','cost','item_volume','bin_volume'):
                if not math.isclose(t[k],float(ref[oid][k]),rel_tol=1e-10,abs_tol=1e-6):
                    raise ValueError(f'{name}/{oid}: {k} differs from recorded thesis run: {t[k]} vs {ref[oid][k]}')
            totals.append(t); bins.extend(b); pairs.extend(i)
            if n%100==0 or n==len(selected):
                print(f'{name}: {n}/{len(selected)} orders; validity and reference checks passed',flush=True)
        write_csv(args.output/(name+'_per_order.csv'), totals)
        write_csv(args.output/(name+'_per_bin.csv'), bins)
        write_csv(args.output/(name+'_per_item.csv'), pairs)
        unused = sum(r['unused_volume'] for r in bins)
        summary_rows.append(dict(heuristic=name,orders=len(selected),reference_match=True,**summarise(bins,unused)))
        for group_by in ('bin_type','occupancy'):
            groups = defaultdict(list)
            for r in bins:
                key = r['bin_type'] if group_by=='bin_type' else ('single_item' if r['items']==1 else 'multiple_items')
                groups[key].append(r)
            for key, rows in groups.items():
                grouped_rows.append(dict(heuristic=name,group_by=group_by,group=key,**summarise(rows,unused)))
        ratios = sorted(r['item_to_smallest_feasible_bin_pct'] for r in pairs)
        item_summary.append(dict(heuristic=name,items=len(pairs),
                                 mean_item_to_smallest_bin_pct=sum(ratios)/len(ratios),
                                 median_item_to_smallest_bin_pct=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2,
                                 items_below_50_pct=100*sum(r<50 for r in ratios)/len(ratios)))
    write_csv(args.output/'overall_summary.csv',summary_rows)
    write_csv(args.output/'grouped_summary.csv',grouped_rows)
    write_csv(args.output/'item_catalogue_summary.csv',item_summary)
    for file in (args.dataset,args.splits,Path(__file__).resolve().parent/'bin_types.py'):
        metadata['source_sha256'][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Complete: {args.output.resolve()}')
    print('Utilisation uses total item volume / total opened-bin volume. These are descriptive diagnostics, not proof that unused space is unavoidable.')


if __name__=='__main__':
    main()
