"""Reject credential-like files and recognizable secrets without printing values."""
import argparse
import re
from pathlib import Path

RULES = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'OpenAI-style key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
    'cloud access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'Google API key': re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    'Slack token': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
    'credential URL': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
}
ALLOWED_ROOTS = {'assets','data','docs','index.html','README.md','.nojekyll','.gitattributes'}

def inspect(root):
    root=Path(root).resolve()
    findings=[]
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if '.git' in rel.parts:continue
        if path.is_symlink():
            findings.append((str(rel),'symbolic link'));continue
        if not path.is_file():continue
        if rel.parts[0] not in ALLOWED_ROOTS:
            findings.append((str(rel),'unexpected export path'))
        name=path.name.lower()
        if name.startswith('.env') or name in {'.netrc','.npmrc','credentials.json','id_rsa','id_ed25519'} or path.suffix.lower() in {'.pem','.pfx','.p12','.key'}:
            findings.append((str(rel),'credential-like filename'))
        if path.suffix.lower()=='.pdf':
            from pypdf import PdfReader
            text='\n'.join(p.extract_text() or '' for p in PdfReader(path).pages)
        elif path.suffix.lower() in {'.gif','.png','.jpg','.webp'}:continue
        else:text=path.read_text(encoding='utf-8',errors='replace')
        for label,pattern in RULES.items():
            if pattern.search(text):findings.append((str(rel),label))
    return findings

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    findings=inspect(args.directory)
    for path,label in findings:print(f'BLOCKED: {path} ({label}; value redacted)')
    if findings:raise SystemExit(1)
    print('Publish allowlist and recognizable-secret checks passed.')

if __name__=='__main__':main()
