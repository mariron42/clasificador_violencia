"""Aggregate public task annotations. Never export report text or individual IDs."""
from pathlib import Path
import csv, io, json, hashlib, zipfile

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
parser.add_argument('--output-dir',type=Path,default=Path('outputs/annotations'))
args=parser.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
archive=args.archive
labels = ['Economic', 'Physical', 'Property-related', 'Psychological', 'Sexual', 'Vicarious', 'N/A']
result = {'source': 'Organizer-provided starting kit', 'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'splits': {}}
with zipfile.ZipFile(archive) as z:
    for split, softname in [('train','trainSoft'), ('devel','develSoft')]:
        member = f'subtask2/SoftLabels/{softname}.csv'
        rows = list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))
        hard = {r['ID']: r for r in csv.DictReader(io.StringIO(z.read(f'subtask2/{split}.csv').decode('utf-8-sig')))}
        stats, mismatch = [], 0
        for j, label in enumerate(labels):
            ps = [float(r[f'L{j}']) for r in rows]
            assert all(abs(p*3-round(p*3)) < 1e-9 for p in ps)
            counts = [sum(round(p*3)==k for p in ps) for k in range(4)]
            label_mismatch = sum(int(float(hard[r['ID']][f'L{j}'])) != int(p > .66) for r,p in zip(rows,ps))
            mismatch += label_mismatch
            stats.append({'column': f'L{j}', 'label': label, 'vote_counts_0_1_2_3': counts,
                          'split_decisions': counts[1]+counts[2], 'split_pct': 100*(counts[1]+counts[2])/len(ps),
                          'mean_population_vote_variance': sum(p*(1-p) for p in ps)/len(ps), 'hard_soft_mismatches': label_mismatch})
        any6 = sum(any(0<float(r[f'L{j}'])<1 for j in range(6)) for r in rows)
        result['splits'][split] = {'n':len(rows), 'labels':stats, 'any_split_six_types':any6,
            'any_split_six_types_pct': 100*any6/len(rows), 'hard_soft_mismatches': mismatch}
    s1=list(csv.DictReader(io.StringIO(z.read('subtask1/train.csv').decode('utf-8-sig'))))
    result['s1_train']={'n':len(s1),'counts':{c:sum(r['CLASS']==c for r in s1) for c in sorted(set(r['CLASS'] for r in s1))}}
result['method'] = 'Per report/type p is the mean of three binary votes. Split vote iff 0 < p < 1. Within-item population variance = p(1-p), equal to 2/9 for split votes and 0 for unanimity. Not a variance between identified workers, uncertainty interval, or severity disagreement metric.'
(OUT/'annotation_consensus.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Aggregate annotation analysis written; no texts or IDs exported.')
