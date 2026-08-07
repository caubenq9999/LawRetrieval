# -*- coding: utf-8 -*-
import json, os, re, sys, statistics, collections, hashlib

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'd:/DSC2026/LegalIR - Public Test-20260806T081424Z-1-001/LegalIR - Public Test'
CTX = os.path.join(BASE, 'selected-contexts', 'selected-contexts')

docs = {}
keysets = collections.Counter()
for fn in os.listdir(CTX):
    if not fn.startswith('context_'):
        continue
    with open(os.path.join(CTX, fn), encoding='utf-8-sig') as f:
        d = json.load(f)
    keysets[tuple(sorted(d.keys()))] += 1
    docs[fn[8:-5]] = d

print(f'Tong so van ban: {len(docs)}')
print('Schema xuat hien:', dict(keysets))

# id trong file co khop ten file khong
mismatch = [k for k, d in docs.items() if 'id' in d and str(d['id']) != k]
print(f'id trong file khac ten file: {len(mismatch)}')

def pct(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p / 100))]

chars = [len(d['passage']) for d in docs.values()]
words = [len(d['passage'].split()) for d in docs.values()]

print('\n=== DO DAI PASSAGE ===')
for name, vals in (('ky tu', chars), ('tu (whitespace)', words)):
    print(f'{name:>16}: min {min(vals)}  p25 {pct(vals,25)}  median {pct(vals,50)}  '
          f'p75 {pct(vals,75)}  p90 {pct(vals,90)}  p95 {pct(vals,95)}  p99 {pct(vals,99)}  max {max(vals)}')
    print(f'{"":>16}  mean {statistics.mean(vals):.0f}  tong {sum(vals):,}')

buckets = [(0,128),(128,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,10**9)]
print('\nPhan bo so tu:')
for lo, hi in buckets:
    n = sum(1 for w in words if lo <= w < hi)
    label = f'{lo}-{hi}' if hi < 10**9 else f'>{lo}'
    print(f'  {label:>12} tu : {n:>5} van ban ({n/len(words)*100:5.1f}%)')

# uoc luong token: tieng Viet ~1.6-2.0 subword/tu voi tokenizer multilingual
print(f'\nUoc luong token (x1.8): median {pct(words,50)*1.8:.0f}, p90 {pct(words,90)*1.8:.0f}, p99 {pct(words,99)*1.8:.0f}')
print(f'So van ban vuot 512 token (~284 tu): {sum(1 for w in words if w > 284)} '
      f'({sum(1 for w in words if w > 284)/len(words)*100:.1f}%)')
print(f'So van ban vuot 256 token (~142 tu): {sum(1 for w in words if w > 142)} '
      f'({sum(1 for w in words if w > 142)/len(words)*100:.1f}%)')

print('\n=== CAU TRUC VAN BAN PHAP LY ===')
pats = {
    'Dieu N':      re.compile(r'(?m)^\s*Điều\s+\d+', re.I),
    'Chuong':      re.compile(r'(?m)^\s*Chương\s+[IVXLC\d]', re.I),
    'Muc':         re.compile(r'(?m)^\s*Mục\s+\d+', re.I),
    'Khoan N.':    re.compile(r'(?m)^\s*\d+\.\s'),
    'Diem a)':     re.compile(r'(?m)^\s*[a-zđ]\)\s'),
}
for name, p in pats.items():
    has = sum(1 for d in docs.values() if p.search(d['passage']))
    tot = sum(len(p.findall(d['passage'])) for d in docs.values())
    print(f'  {name:<12}: {has:>5} van ban co ({has/len(docs)*100:5.1f}%), tong {tot:,} lan xuat hien')

dieu_counts = [len(pats['Dieu N'].findall(d['passage'])) for d in docs.values()]
nz = [c for c in dieu_counts if c]
if nz:
    print(f'  So "Dieu" moi van ban (chi tinh van ban co): median {pct(nz,50)}, p90 {pct(nz,90)}, max {max(nz)}')

print('\n=== NHIEU / TRUNG LAP ===')
h = collections.Counter(hashlib.md5(d['passage'].encode()).hexdigest() for d in docs.values())
dup = sum(c - 1 for c in h.values() if c > 1)
print(f'  Passage trung y het: {dup} ban sao ({len(h)} passage duy nhat)')
crlf = sum(1 for d in docs.values() if '\r\n' in d['passage'])
print(f'  Van ban chua \\r\\n (xuong dong gia trong bang HTML): {crlf} ({crlf/len(docs)*100:.1f}%)')
blank = sum(d['passage'].count('\n\n') for d in docs.values())
print(f'  Tong so dong trong "\\n\\n": {blank:,} (trung binh {blank/len(docs):.0f}/van ban)')

print('\n=== LINK / CHU DE ===')
cats = collections.Counter()
kinds = collections.Counter()
for d in docs.values():
    m = re.match(r'https?://[^/]+/([^/]+)/([^/]+)/', d.get('link', ''))
    if m:
        kinds[m.group(1)] += 1
        cats[m.group(2)] += 1
print('  Loai:', dict(kinds.most_common()))
print('  Top 15 linh vuc:')
for c, n in cats.most_common(15):
    print(f'    {n:>5}  {c}')

print('\n=== QUAN HE VOI CAU HOI ===')
train = json.load(open(os.path.join(BASE, 'train.json'), encoding='utf-8-sig'))
pub = json.load(open(os.path.join(BASE, 'public-official.json'), encoding='utf-8-sig'))
ans_ids = [str(d) for v in train.values() for d in v['answer']]
uniq_ans = set(ans_ids)
print(f'  Cau hoi train: {len(train)}, public: {len(pub)}')
print(f'  Van ban tung la dap an: {len(uniq_ans)}/{len(docs)} ({len(uniq_ans)/len(docs)*100:.1f}%)')
print(f'  Van ban CHUA BAO GIO la dap an: {len(docs)-len(uniq_ans)} -> nhieu kha nang la distractor + dap an cua public/private')
freq = collections.Counter(ans_ids)
print(f'  So lan mot van ban duoc dung lam dap an: median {pct(list(freq.values()),50)}, '
      f'max {max(freq.values())}')
print(f'  Top 5 van ban hay la dap an: {freq.most_common(5)}')

aw = [len(docs[i]['passage'].split()) for i in uniq_ans if i in docs]
print(f'  Do dai van ban LA dap an   : median {pct(aw,50)} tu, p90 {pct(aw,90)}')
ow = [len(docs[i]['passage'].split()) for i in docs if i not in uniq_ans]
print(f'  Do dai van ban KHONG la dap an: median {pct(ow,50)} tu, p90 {pct(ow,90)}')

qw = [len(v['question'].split()) for v in train.values()]
print(f'  Do dai cau hoi: median {pct(qw,50)} tu, p90 {pct(qw,90)}, max {max(qw)}')

print('\n=== VI DU 3 VAN BAN NGAN NHAT ===')
for i in sorted(docs, key=lambda k: len(docs[k]['passage']))[:3]:
    print(f'  [{i}] {len(docs[i]["passage"])} ky tu: {docs[i]["passage"][:200]!r}')
