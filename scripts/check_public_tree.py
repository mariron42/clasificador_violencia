"""Conservative release checks. Findings show locations, never matching values."""
from pathlib import Path
import re, zipfile, json

ROOT=Path(__file__).resolve().parents[1]
rules={
    'private_home':r'/(?:lustre/home|home)/[A-Za-z0-9_]+/',
    'windows_home':r'[A-Z]:[\\/]+Users[\\/]+',
    'personal_email':r'[\w.+-]+@(?:gmail|hotmail|outlook)\.com',
    'private_key':r'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',
    'credential':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,})',
    'cluster_account':r'estudiante[_-]\d+',
    'private_network':r'\b(?:192\.168\.\d+\.\d+|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b',
}
findings=[];checked=0
for p in ROOT.rglob('*'):
    if not p.is_file() or any(v in p.parts for v in ['.git','__pycache__','.venv']):continue
    rel=p.relative_to(ROOT).as_posix()
    if p.resolve()==Path(__file__).resolve():continue
    if p.suffix.lower() in ['.csv','.jsonl','.zip','.pt','.safetensors','.pem','.key']:
        findings.append({'file':rel,'rule':'prohibited_release_file'});continue
    chunks=[]
    if p.suffix=='.pptx':
        with zipfile.ZipFile(p) as z:
            chunks=[(n,z.read(n).decode('utf8',errors='replace')) for n in z.namelist() if n.endswith(('.xml','.rels'))]
    elif p.suffix=='.pdf':
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit('Install pypdf to inspect PDF release artifacts.')
        r=PdfReader(p);chunks=[('text','\n'.join(x.extract_text() for x in r.pages)),('metadata',str(r.metadata))]
    else:
        try:chunks=[('',p.read_text(encoding='utf8'))]
        except UnicodeDecodeError:
            findings.append({'file':rel,'rule':'unreviewed_binary'});continue
    checked+=1
    for member,body in chunks:
        for name,pattern in rules.items():
            if re.search(pattern,body,re.I):findings.append({'file':rel,'member':member,'rule':name})
print(json.dumps({'checked_files':checked,'findings':findings},indent=2))
raise SystemExit(bool(findings))
