# -*- coding: utf-8 -*-
import json, os, re, sys, statistics, collections

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'd:/DSC2026/LegalIR - Public Test-20260806T081424Z-1-001/LegalIR - Public Test'
CTX = os.path.join(BASE, 'selected-contexts', 'selected-contexts')

docs = {}
for fn in os.listdir(CTX):
    if fn.startswith('context_'):
        with open(os.path.join(CTX, fn), encoding='utf-8-sig') as f:
            docs[fn[8:-5]] = json.load(f)

train = json.load(open(os.path.join(BASE, 'train.json'), encoding='utf-8-sig'))
ans_ids = {str(d) for v in train.values() for d in v['answer']}

def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p / 100))]

print('=== VAN BAN RONG / QUA NGAN ===')
for thr in (0, 50, 200):
    bad = [k for k, d in docs.items() if len(d['passage'].split()) <= thr]
    hit = [k for k in bad if k in ans_ids]
    print(f'  <= {thr:>3} tu: {len(bad):>3} van ban, trong do {len(hit)} tung la dap an train {hit[:8]}')

print('\n=== HIEU QUA LAM SACH ===')
def clean(t):
    t = t.replace('\r', '')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{2,}', '\n', t)
    return t.strip()

sample = list(docs.values())[:2000]
raw = sum(len(d['passage']) for d in sample)
cln = sum(len(clean(d['passage'])) for d in sample)
print(f'  Tren 2000 van ban: {raw:,} -> {cln:,} ky tu (giam {100*(raw-cln)/raw:.1f}%)')

print('\n=== CHUNK THEO "DIEU" ===')
DIEU = re.compile(r'(?m)^\s*Điều\s+\d+')
per_doc, sizes, no_dieu_words = [], [], []
for k, d in docs.items():
    t = clean(d['passage'])
    pos = [m.start() for m in DIEU.finditer(t)]
    if not pos:
        no_dieu_words.append(len(t.split()))
        continue
    pos.append(len(t))
    parts = [t[pos[i]:pos[i+1]] for i in range(len(pos)-1)]
    head = t[:pos[0]]
    if len(head.split()) > 30:
        parts.insert(0, head)
    per_doc.append(len(parts))
    sizes.extend(len(p.split()) for p in parts)

print(f'  Van ban tach duoc theo Dieu: {len(per_doc)}/{len(docs)}')
print(f'  So chunk/van ban : median {pct(per_doc,50)}, p90 {pct(per_doc,90)}, max {max(per_doc)}')
print(f'  Tong so chunk    : {len(sizes):,}')
print(f'  Do dai chunk (tu): median {pct(sizes,50)}, p75 {pct(sizes,75)}, p90 {pct(sizes,90)}, '
      f'p95 {pct(sizes,95)}, p99 {pct(sizes,99)}, max {max(sizes)}')
for lo, hi in ((0,32),(32,128),(128,284),(284,568),(568,10**9)):
    n = sum(1 for s in sizes if lo <= s < hi)
    lab = f'{lo}-{hi}' if hi < 10**9 else f'>{lo}'
    print(f'    {lab:>9} tu: {n:>7,} chunk ({n/len(sizes)*100:5.1f}%)')
print(f'  Van ban KHONG co "Dieu": {len(no_dieu_words)}, do dai median {pct(no_dieu_words,50) if no_dieu_words else 0} tu')

print('\n=== UOC LUONG CHI PHI EMBED ===')
tot_words = sum(len(clean(d['passage']).split()) for d in docs.values())
print(f'  Tong so tu sau lam sach: {tot_words:,}  (~{tot_words*1.8/1e6:.0f}M token)')
for win in (128, 256, 400):
    n = tot_words / (win * 0.75)   # stride 75% = overlap 25%
    print(f'  Chunk co dinh {win:>3} tu (overlap 25%): ~{n:,.0f} chunk')

print('\n=== TIN HIEU LEXICAL: cau hoi vs van ban dap an ===')
STOP = set('la cua va co the cho khong duoc trong voi nhu tai theo mot nao gi ra thi ma o den tu bao nhieu nhung se da'.split())
def toks(s):
    return [w for w in re.findall(r'\w+', s.lower()) if w not in STOP and len(w) > 1]

import random
random.seed(0)
qs = random.sample(list(train.items()), 300)
cov_gold, cov_rand = [], []
allids = list(docs)
for qid, v in qs:
    q = set(toks(v['question']))
    if not q:
        continue
    gold = str(v['answer'][0])
    if gold in docs:
        gt = set(toks(clean(docs[gold]['passage'])))
        cov_gold.append(len(q & gt) / len(q))
    rid = random.choice(allids)
    rt = set(toks(clean(docs[rid]['passage'])))
    cov_rand.append(len(q & rt) / len(q))

print(f'  Ty le tu khoa cau hoi xuat hien trong van ban DUNG   : '
      f'mean {statistics.mean(cov_gold):.3f}, median {pct(cov_gold,50):.3f}')
print(f'  Ty le tu khoa cau hoi xuat hien trong van ban NGAU NHIEN: '
      f'mean {statistics.mean(cov_rand):.3f}, median {pct(cov_rand,50):.3f}')
print(f'  Cau hoi co >=90% tu khoa nam trong van ban dung: '
      f'{100*sum(1 for c in cov_gold if c>=0.9)/len(cov_gold):.1f}%')
