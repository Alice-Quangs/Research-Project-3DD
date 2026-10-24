"""Run from Main after installing. Tests orientation rule and basic state/geometry."""
import importlib
from itertools import combinations
from run_stage2 import combine

def main():
    catalogue=[dict(name='test',length=10,width=10,height=10,cost=1,max_weight=99)]
    for h in ['ha1','ha2','ha3']:
        m=importlib.import_module('heuristics.'+h); bins=[]
        item=dict(length=2,width=3,height=4)
        b=m.pack_item(catalogue,bins,2,3,4)
        assert b.placed_items[0]==(0,0,0,4,3,2),(h,b.placed_items)
        assert b.used_volume==24
        before=repr(b.__dict__);assert m.can_pack(b,item);assert repr(b.__dict__)==before
        for _ in range(5):m.pack_item(catalogue,bins,2,3,4)
        assert sum(len(b.placed_items) for b in bins)==6
        assert sum(b.used_volume for b in bins)==144
        for b in bins:
            for a in b.placed_items:
                assert all(0<=a[i]<a[i+3]<=10 for i in range(3))
            for a,c in combinations(b.placed_items,2):
                assert any(a[i+3]<=c[i] or c[i+3]<=a[i] for i in range(3))
        print(h,'orientation, state and geometry checks passed')
    sources=[(str(i),[dict(id=f'{i}-{j}') for j in range(3)]) for i in range(5)]
    a,meta=combine(sources,[4,8],3,2026)
    assert (a,meta)==combine(sources,[4,8],3,2026)
    lookup=dict(sources)
    for (oid,items),r in zip(a,meta):
        assert len(r['source_order_ids'])==len(set(r['source_order_ids']))
        assert items==[x for source in r['source_order_ids'] for x in lookup[source]]
        assert len(items)>=r['target_items']
    print('Combined-order construction checks passed')
if __name__=='__main__':main()
