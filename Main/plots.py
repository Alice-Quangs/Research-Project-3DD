"""Purple/violet research figures for the shared packing framework.

Dependencies: matplotlib, numpy, scikit-learn (classification curves only).
No plotting or global style changes occur on import. All functions return a
Matplotlib Figure; callers may customise it, save it, then plt.close(fig).

CLI (reads benchmark.py outputs; does not rerun experiments):
  python plots.py --results results/ha1/RUN_DIRECTORY
  python plots.py --results results/ha1/RUN_DIRECTORY \
      --xgb-importance results/xgb_importance.json \
      --nn-importance results/nn_importance.json

Importance JSON is the unchanged result of metrics.permutation_importance.
Export PNG (300 dpi) and vector PDF. Output defaults to RESULTS/plots.
Validation sweeps are labelled explicitly. Cost is total bin cost; bins is
TOTAL opened bins across orders, not bin types. Runtime bars show the median
whole-run runtime with min/max whiskers (not confidence intervals).

Baseline = lavender, XGBoost = violet, NN = deep plum. Hatches and markers
also distinguish methods for grayscale printing. Axes use dark plum text on
white backgrounds. No values are normalised or rescaled implicitly.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {'baseline':'#B69AD6', 'xgboost':'#8050C8', 'neural_network':'#49206B'}
LABELS = {'baseline':'Baseline', 'xgboost':'XGBoost', 'neural_network':'Neural network'}
HATCHES = {'baseline':'//', 'xgboost':'', 'neural_network':'..'}
MARKERS = {'baseline':'s', 'xgboost':'o', 'neural_network':'D'}
STYLE = {'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
         'axes.labelsize':10, 'axes.edgecolor':'#715B80', 'axes.labelcolor':'#35243F',
         'text.color':'#35243F', 'xtick.color':'#35243F', 'ytick.color':'#35243F',
         'axes.spines.top':False, 'axes.spines.right':False,
         'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
         'pdf.fonttype':42, 'ps.fonttype':42}


def _method(name):
    if name not in COLORS:
        raise ValueError(f'Unknown method: {name}')
    return LABELS[name]


def _number(value):
    if value is None or not np.isfinite(value):
        raise ValueError('Plot values must be finite; undefined metrics cannot be plotted.')
    return float(value)


def _grid(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#EEE7F4', linewidth=.8)


def save_figure(fig, output_stem, formats=('png','pdf')):
    """Save a figure at an extensionless path; return written Paths."""
    stem = Path(output_stem)
    if any(fmt not in ('png','pdf','svg') for fmt in formats):
        raise ValueError('Supported formats: png, pdf, svg.')
    stem.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for fmt in formats:
        path=Path(str(stem)+'.'+fmt)
        fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        paths.append(path)
    return paths


def plot_benchmark_summary(summary, title='Packing comparison'):
    """Plot one fixed filter-start configuration per method, including bin count."""
    summary=list(summary)
    if not summary or len({r['method'] for r in summary})!=len(summary):
        raise ValueError('Provide one summary row per method; use plot_filter_sweep for sweeps.')
    summary=sorted(summary,key=lambda r:list(COLORS).index(r['method']))
    fields=[('median_cost','Total bin cost',False),('median_bins','Number of bins',False),
            ('median_runtime_s','Runtime (seconds)',False),
            ('median_utilisation','Volume utilisation (%)',True)]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,(key,label,percent) in zip(axes.flat,fields):
            values=[_number(r[key])*(100 if percent else 1) for r in summary]
            bars=ax.bar(range(len(summary)),values,
                        color=[COLORS[r['method']] for r in summary],
                        edgecolor='#492F60',linewidth=.7,
                        hatch=[HATCHES[r['method']] for r in summary])
            if key=='median_runtime_s':
                lower=[v-_number(r['min_runtime_s']) for v,r in zip(values,summary)]
                upper=[_number(r['max_runtime_s'])-v for v,r in zip(values,summary)]
                ax.errorbar(range(len(summary)),values,yerr=[lower,upper],fmt='none',
                            color='#35243F',capsize=4)
            ax.set_xticks(range(len(summary)),[_method(r['method']) for r in summary])
            ax.set_ylabel(label);ax.set_ylim(bottom=0);_grid(ax)
            ax.margins(y=.2)
            ax.bar_label(bars,fmt='%.2f' if key in ('median_runtime_s','median_utilisation') else '%g',padding=4)
        fig.suptitle(title+'\nMedians across runs; runtime whiskers show min–max',fontsize=13)
    return fig


def plot_filter_sweep(summary, partition='validation', title='Filter-start experiment'):
    """Runtime, cost and bin count against minimum open bins; baseline reference."""
    if partition!='validation':
        raise ValueError('Filter-start experiments must use validation results.')
    summary=list(summary)
    baseline=[r for r in summary if r['method']=='baseline']
    if len(baseline)>1:
        raise ValueError('Multiple baseline rows supplied.')
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True,layout='constrained')
        for ax,key,label in zip(axes,('median_runtime_s','median_cost','median_bins'),
                               ('Runtime (seconds)','Total bin cost','Number of bins')):
            for method in ('xgboost','neural_network'):
                group=sorted([r for r in summary if r['method']==method],key=lambda r:r['min_bins'])
                if not group:continue
                x=[r['min_bins'] for r in group]
                if len(set(x))!=len(x):raise ValueError('Duplicate sweep settings.')
                ax.plot(x,[_number(r[key]) for r in group],color=COLORS[method],
                        marker=MARKERS[method],label=LABELS[method],linewidth=2)
            if baseline:
                ax.axhline(_number(baseline[0][key]),color=COLORS['baseline'],
                           linestyle='--',label='Baseline',linewidth=2)
            ax.set_ylabel(label);ax.set_ylim(bottom=0);_grid(ax);ax.legend(frameon=False)
        axes[-1].set_xlabel('Minimum open bins before filtering')
        ticks=sorted({r['min_bins'] for r in summary if r['method']!='baseline'})
        if not ticks:raise ValueError('No learned-method sweep results.')
        axes[-1].set_xticks(ticks)
        fig.suptitle(title+' — validation orders',fontsize=13)
    return fig


def plot_feature_importance(results, title='Permutation feature importance', top_n=None):
    """results maps model names to metrics.permutation_importance output.

    Shared ranking and x scale across models. Error bars are permutation SD,
    not confidence intervals. Negative values are preserved.
    """
    if not results:raise ValueError('No importance results supplied.')
    methods=list(results)
    reference=results[methods[0]]
    names=[r['feature'] for r in reference['features']]
    if not names or len(set(names))!=len(names):raise ValueError('Invalid feature names.')
    tables={}
    for method,result in results.items():
        _method(method)
        for key in ('scoring','heuristic','rows','seed','repeats'):
            if result[key]!=reference[key]:raise ValueError(f'Importance results differ in {key}.')
        table={r['feature']:r for r in result['features']}
        if len(table)!=len(result['features']) or set(table)!=set(names):
            raise ValueError('Importance feature schemas differ.')
        tables[method]=table
    ranking=sorted(names,key=lambda name:np.mean([_number(t[name]['importance_mean']) for t in tables.values()]),reverse=True)
    if top_n is not None:
        if type(top_n) is not int or top_n<1:raise ValueError('top_n must be a positive integer.')
        ranking=ranking[:top_n]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,len(methods),figsize=(5*len(methods)+2,max(4,.32*len(ranking)+1.7)),
                              sharey=True,sharex=True,squeeze=False,layout='constrained')
        for ax,method in zip(axes.flat,methods):
            table=tables[method]
            values=[_number(table[n]['importance_mean']) for n in ranking]
            sd=[_number(table[n]['importance_std']) for n in ranking]
            if any(s<0 for s in sd):raise ValueError('Importance SD cannot be negative.')
            ax.barh(range(len(ranking)),values,xerr=sd,color=COLORS[method],
                    edgecolor='#492F60',linewidth=.5,capsize=2,hatch=HATCHES[method])
            ax.axvline(0,color='#715B80',linewidth=.8)
            ax.set_yticks(range(len(ranking)),[n.replace('_',' ') for n in ranking])
            ax.set_title(_method(method));ax.set_xlabel('Decrease in average precision');_grid(ax,'x')
        axes[0,0].invert_yaxis()
        fig.suptitle(title+'\nError bars: standard deviation across permutations',fontsize=13)
    return fig


def plot_classification_curves(y_true, probabilities, title='Held-out classification performance'):
    """Shared labels and aligned probability vectors; plot ROC and PR curves."""
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    from metrics import classification_metrics
    y=np.asarray(y_true)
    if len(np.unique(y))!=2:raise ValueError('Curves require both target classes.')
    if not probabilities:raise ValueError('No probability vectors supplied.')
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for index,(method,p) in enumerate(probabilities.items()):
            classification_metrics(y,p,.5) # validate shape, labels and probability range
            label=_method(method)
            fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
            style='-' if index%2==0 else '--'
            axes[0].plot(fpr,tpr,color=COLORS[method],linestyle=style,label=f'{label} (AUROC {roc_auc_score(y,p):.3f})')
            axes[1].step(recall,precision,where='post',color=COLORS[method],linestyle=style,
                         label=f'{label} (AP {average_precision_score(y,p):.3f})')
        axes[0].plot([0,1],[0,1],color='#B69AD6',linestyle=':',label='Chance')
        axes[1].axhline(float(np.mean(y)),color='#B69AD6',linestyle=':',label='Positive prevalence')
        for ax in axes:
            ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(frameon=False,fontsize=8);_grid(ax)
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall curve')
        fig.suptitle(title,fontsize=13)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--xgb-importance',type=Path)
    parser.add_argument('--nn-importance',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.results/'summary.json').read_text())
    metadata=json.loads((args.results/'metadata.json').read_text())
    output=args.output or args.results/'plots'
    title=f"{metadata['heuristic'].upper()} · {metadata['partition']} orders"
    sweep=len({r['method'] for r in summary})!=len(summary)
    if sweep:
        fig=plot_filter_sweep(summary,metadata['partition'],title)
        stem='filter_start_sweep'
    else:
        fig=plot_benchmark_summary(summary,title);stem='packing_comparison'
    paths=save_figure(fig,output/stem);plt.close(fig)
    importance={}
    for method,path in [('xgboost',args.xgb_importance),('neural_network',args.nn_importance)]:
        if path:
            result=json.loads(path.read_text())
            if result['heuristic']!=metadata['heuristic']:raise ValueError('Importance heuristic differs from benchmark.')
            importance[method]=result
    if importance:
        fig=plot_feature_importance(importance,title+' · feature importance')
        paths+=save_figure(fig,output/'feature_importance');plt.close(fig)
    for path in paths:print(path)


if __name__=='__main__':
    main()
