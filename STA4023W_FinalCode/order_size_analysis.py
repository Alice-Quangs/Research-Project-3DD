"""Mean packing results and paired runtime versus order size (Python 3.10+).

Install: python -m pip install pandas numpy matplotlib
Existing results:
  python order_size_analysis.py --input results/ha1/RUN
Fresh benchmark (arguments after -- go unchanged to your benchmark.py):
  python order_size_analysis.py --input results/ha1/size_run --run -- --heuristic ha1 --methods baseline xgboost neural_network --repetitions 7 [your usual model/threshold/module arguments]
Run separately for ha1, ha2 and ha3, using their own trained models/configuration.
The runner adds --output itself. Existing result folders are never overwritten.

Compatible with the saved shared benchmark.py's per_order.csv schema. This
script does not change packing, filters, thresholds, caching or timing. Loading
models is excluded by that benchmark; prediction/features/fallbacks are included.
Use fixed validation-selected settings and the same test orders/item sequences.
Run timing experiments sequentially with other heavy jobs stopped. Seven or more
repetitions help quantify timing noise. Existing CSVs need no rerun.

Mean across runs of TOTAL cost/bins/runtime is distinct from mean per order.
Utilisation is volume-weighted within each run, then averaged across runs.
Order-size results first average repeats for each order; each order has equal
weight. Speedup is ratio of mean runtimes, NOT mean of individual speedups.
Bootstrap intervals resample paired orders, not repeated timings; they describe
between-order uncertainty, not independent timing trials. Crossover is exploratory:
inspect sample counts/intervals, and do not retune on the test set.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyse(folder, width=25, bootstrap=2000):
    d = pd.read_csv(folder / 'per_order.csv', dtype={'order_id': str})
    required = ['method','repetition','order_id','items','bins','cost','runtime_s','item_volume','bin_volume']
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. Supply the current per_order.csv schema.')
    for c in required[3:]:
        d[c] = pd.to_numeric(d[c], errors='raise')
        if not np.isfinite(d[c]).all() or (d[c] < 0).any():
            raise ValueError(f'Invalid numeric values in {c}')
    if (d.runtime_s <= 0).any() or (d.bin_volume <= 0).any() or (d['items'] < 1).any():
        raise ValueError('Runtime, capacity and item count must be positive.')
    if (d['items'] % 1 != 0).any():
        raise ValueError('Item count must be integral.')
    # One setting per method: do not silently mix validation sweeps.
    for method, g in d.groupby('method'):
        for c in ['min_bins','threshold']:
            if c in g and g[c].nunique(dropna=False) != 1:
                raise ValueError(f'Multiple {c} settings for {method}; analyse separately.')
    if d.duplicated(['method','repetition','order_id']).any():
        raise ValueError('Duplicate method/repetition/order rows.')
    b = d[d.method == 'baseline']
    if b.empty:
        raise ValueError('Baseline rows required.')
    keys = set(zip(b.repetition, b.order_id))
    order_set = set(b.order_id)
    for _, g in b.groupby('repetition'):
        if set(g.order_id) != order_set:
            raise ValueError('Baseline repetitions must contain the same orders.')
    for method, g in d.groupby('method'):
        if set(zip(g.repetition,g.order_id)) != keys:
            raise ValueError(f'{method} has unmatched orders/repetitions.')
    if (d.groupby('order_id')['items'].nunique() != 1).any():
        raise ValueError('Item counts differ for the same order.')
    out = folder / 'mean_order_size_analysis'
    out.mkdir(exist_ok=False)
    metrics = ['runtime_s','cost','bins','items','item_volume','bin_volume']
    optional = [c for c in ['filter_activations','empty_shortlist_fallbacks','available_candidates','shortlisted_candidates','raw_model_rejections'] if c in d]
    runs = d.groupby(['method','repetition'])[metrics + optional].sum().reset_index()
    runs['utilisation_pct'] = 100*runs.item_volume/runs.bin_volume
    runs.to_csv(out/'run_totals.csv',index=False)
    summary = runs.groupby('method')[['runtime_s','cost','bins','utilisation_pct']].agg(['mean','std','min','max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['speedup_vs_baseline'] = summary.loc['baseline','runtime_s_mean']/summary.runtime_s_mean
    summary['cost_change_pct'] = 100*(summary.cost_mean/summary.loc['baseline','cost_mean']-1)
    summary.to_csv(out/'mean_summary.csv')
    orders = d.groupby(['method','order_id']).agg({**{c:'mean' for c in metrics+optional},'repetition':'count'}).reset_index().rename(columns={'repetition':'n_repetitions'})
    base = orders[orders.method=='baseline'][['order_id','runtime_s','cost','bins']].rename(columns={c:'baseline_'+c for c in ['runtime_s','cost','bins']})
    orders = orders.merge(base,on='order_id',validate='many_to_one')
    orders['speedup'] = orders.baseline_runtime_s/orders.runtime_s
    orders['runtime_difference_s'] = orders.runtime_s-orders.baseline_runtime_s
    orders['cost_change_pct'] = 100*(orders.cost/orders.baseline_cost.replace(0,np.nan)-1)
    orders['bins_difference'] = orders.bins-orders.baseline_bins
    orders['size_lower'] = ((orders['items']-1)//width*width+1).astype(int)
    orders['size_upper'] = orders.size_lower+width-1
    orders.to_csv(out/'per_order_means.csv',index=False)
    rng = np.random.default_rng(2026)
    grouped=[]
    for (method,lo,hi),g in orders.groupby(['method','size_lower','size_upper']):
        a=g.baseline_runtime_s.to_numpy(); z=g.runtime_s.to_numpy()
        intervals=[]
        if len(g)>1:
            for _ in range(bootstrap):
                ix=rng.integers(0,len(g),len(g))
                intervals.append(a[ix].mean()/z[ix].mean())
        low,high=np.quantile(intervals,[.025,.975]) if intervals else (np.nan,np.nan)
        row=dict(method=method,size_lower=lo,size_upper=hi,n_orders=len(g),mean_items=g['items'].mean(),mean_runtime_s=z.mean(),baseline_mean_runtime_s=a.mean(),speedup=a.mean()/z.mean(),speedup_ci_low=low,speedup_ci_high=high,faster_order_pct=100*(z<a).mean(),mean_cost_change_pct=g.cost_change_pct.mean(),mean_bins_difference=g.bins_difference.mean())
        for c in optional:
            row['mean_'+c]=g[c].mean()
        grouped.append(row)
    sizes=pd.DataFrame(grouped)
    sizes.to_csv(out/'by_order_size.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,metric,label in zip(axs.flat,['cost','bins','runtime_s','utilisation_pct'],['Total bin cost','Total bins','Total runtime (s)','Volume utilisation (%)']):
        ax.bar(summary.index,summary[metric+'_mean'],yerr=summary[metric+'_std'].fillna(0),capsize=4,color=['#ae90cb','#8150bd','#422367'][:len(summary)])
        ax.set_ylabel(label);ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Arithmetic means across runs; whiskers = one standard deviation')
    fig.tight_layout();fig.savefig(out/'packing_means.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for method,g in sizes.groupby('method'):
        axs[0].plot(g.mean_items,g.mean_runtime_s,'o-',label=method)
        if method!='baseline':
            axs[1].plot(g.mean_items,g.speedup,'o-',label=method)
            axs[1].fill_between(g.mean_items,g.speedup_ci_low,g.speedup_ci_high,alpha=.15)
    axs[0].set_ylabel('Mean runtime per order (s)')
    axs[1].set_ylabel('Baseline mean time / model mean time')
    axs[1].axhline(1,color='black',linestyle='--')
    for ax in axs:
        ax.set_xlabel(f'Mean items in order (bands of {width})');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Runtime versus order size; shading = paired-order bootstrap 95% interval')
    fig.tight_layout();fig.savefig(out/'runtime_by_order_size.png',dpi=180);plt.close(fig)
    notes = ['Speedup > 1 means faster than baseline. Band intervals resample paired orders.',
             'A single faster band is not evidence of a persistent crossover.',
             'Inspect counts, neighbouring bands, packing quality and filter activity.',
             'No extrapolation beyond the observed order sizes is justified.',
             'Repeated timing rows are averaged within order before size analysis.',
             'Model loading, cache policy and timing scope are inherited from benchmark.py.',
             'Order size is a proxy for candidate-search work; it does not prove causation.']
    (out/'interpretation.txt').write_text('\n'.join(notes),encoding='utf-8')
    print(summary.to_string());print(f'\nSaved analysis: {out}')


def main():
    argv=sys.argv[1:]
    split=argv.index('--') if '--' in argv else len(argv)
    own,forward=argv[:split],argv[split+1:]
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--benchmark',type=Path,default=Path('benchmark.py'))
    p.add_argument('--bin-width',type=int,default=25)
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args(own)
    if args.bin_width<1 or args.bootstrap<100:
        p.error('Use positive bin width and at least 100 bootstrap resamples.')
    if args.run:
        if args.input.exists(): p.error('Fresh run needs a new --input directory.')
        if any(x=='--output' or x.startswith('--output=') for x in forward):
            p.error('Use --input to set benchmark output, not --output.')
        subprocess.run([sys.executable,str(args.benchmark),*forward,'--output',str(args.input)],check=True)
    elif forward:
        p.error('Arguments after -- require --run.')
    analyse(args.input,args.bin_width,args.bootstrap)

if __name__=='__main__':
    main()
