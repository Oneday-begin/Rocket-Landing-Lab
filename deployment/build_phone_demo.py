"""Export the existing lab into a portable, read-only interview demo."""
import sys, json, shutil, sqlite3, zlib, argparse
from pathlib import Path
import hashlib

parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
ROOT=args.project_root.resolve()
OUT=(args.output or ROOT/'.cache/phone-demo/export').resolve()
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from rocket_landing.predictor import features

OUT.mkdir(parents=True,exist_ok=True)
(OUT/'data/flights').mkdir(parents=True,exist_ok=True)
(OUT/'assets').mkdir(exist_ok=True)
(OUT/'docs').mkdir(exist_ok=True)
def save(path,value):
    (OUT/path).write_text(json.dumps(value,separators=(',',':'),allow_nan=False),encoding='utf-8')

db=sqlite3.connect('file:'+str(ROOT/'artifacts/lab/flights.sqlite').replace('\\','/')+'?mode=ro',uri=True)
db.row_factory=sqlite3.Row
records=[dict(r) for r in db.execute('SELECT * FROM flights ORDER BY model,scenario,seed')]
db.close()
model_path=ROOT/'artifacts/lab/predictor.joblib'
report=json.loads((ROOT/'artifacts/lab/predictor-report.json').read_text())
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==report['model_sha256']
saved=joblib.load(model_path)
summaries=[]
for r in records:
    samples=json.loads(zlib.decompress(r.pop('telemetry')))
    summaries.append({k:r[k] for k in ['id','model','scenario','seed','model_sha256','outcome','score','steps','fuel','hover']})
    if r['steps']>150:
        vector=features(samples).reshape(1,-1)
        probs=saved['model'].predict_proba(vector)[0]
        r['forecast']={'available':True,'at_step':150,'split':saved['assignments'].get(r['id'],'new flight'),
            'probabilities':dict(zip(saved['model'].classes_,map(float,probs))),
            'prediction':str(saved['model'].predict(vector)[0]),'outcome':r['outcome']}
    else:r['forecast']={'available':False,'reason':'Flight ended by three seconds; no early forecast.'}
    # Keep each original step for a faithful 50 Hz replay; compact numeric representation.
    r['samples']=[[int(s[0]),*[round(float(v),5) for v in s[1:9]],int(s[9]),round(float(s[10]),5)] for s in samples]
    save('data/flights/'+r['id']+'.json',r)
save('data/flights.json',summaries)
save('data/predictor.json',{'available':True,**report})
shutil.copy2(ROOT/'artifacts/lab/broader-results.json',OUT/'data/training.json')
shutil.copy2(ROOT/'docs/Rocket-Landing-Placement-Report.pdf',OUT/'docs/Rocket-Landing-Placement-Report.pdf')
for name in ['baseline','hover']:
    gif=next((ROOT/f'runs/{name}/seed_0/videos').glob('*/*.gif'))
    shutil.copy2(gif,OUT/f'assets/{name}.gif')
documents=[]
for filename in ['README.md','docs/WEB-LAB.md']:
    words=(ROOT/filename).read_text(encoding='utf-8').split()
    for i in range(0,len(words),150):
        documents.append({'id':'S'+str(len(documents)+1),'title':filename,'path':filename,'text':' '.join(words[i:i+210])})
save('data/evidence.json',documents)
html=(ROOT/'rocket_landing/web/index.html').read_text(encoding='utf-8')
html=html.replace('href="/assets/','href="assets/').replace('src="/assets/','src="assets/').replace('href="/docs/','href="docs/')
html=html.replace('href="/lab','href="?lab=1').replace('href="/"','href="./"')
html=html.replace('src="/api/demo/baseline"','src="assets/baseline.gif"').replace('src="/api/demo/hover"','src="assets/hover.gif"')
html=html.replace('<script src="assets/app.js" defer>','<script src="assets/demo-api.js" defer></script><script src="assets/app.js" defer>')
html=html.replace('● Simulation data','● Saved flight evidence').replace('Your models. Your flight data.','720 measured simulator flights.')
html=html.replace('● Local workspace','● Interview demo').replace('Local workspace','Interview demo')
html=html.replace('A classifier predicts the final outcome from only the first three seconds.','Saved classifier forecasts use only the first three seconds of each recorded flight.')
html=html.replace('Ask questions about the project. Get calculations or locally generated answers with sources.','Search project evidence and check exact calculations from saved flights.')
html=html.replace('Get calculations or locally generated answers with sources.','Get evidence excerpts and exact flight-data calculations.')
html=html.replace('Ask about the methods or results. Every response comes with its evidence.','Search the project evidence and measured results. This online demo shows excerpts and calculations; local AI generation runs in the desktop lab.')
html=html.replace('Numbers are computed from flight records; explanations retrieve project evidence before local generation.','Numbers are computed from saved flights. Other questions search project evidence. Local language generation is available in the desktop lab.')
html=html.replace('Generated answers can make mistakes. Expand the cited excerpts to check support. If Ollama is unavailable, the assistant shows retrieved excerpts instead.','This hosted demo uses evidence search, without a running language model. Expand each source excerpt to inspect support. The full local app also supports Ollama generation.')
html=html.replace('Runs locally and can take several minutes. A new seed folder preserves earlier models.','The completed experiment is shown below. Run new training from the full Python app on your laptop.')
html=html.replace('Fine-tune a copy of the fixed policy with changing wind and turbulence.','Inspect a completed fine-tuning experiment with changing wind and turbulence.')
(OUT/'index.html').write_text(html,encoding='utf-8')
js=(ROOT/'rocket_landing/web/app.js').read_text(encoding='utf-8')
js=js.replace('async function api(','async function networkApi(')
js=js.replace('setInterval(pollJob,2500);pollJob()','')
js=js.replace("location.pathname==='/lab'","new URLSearchParams(location.search).has('lab')")
js=js.replace("$('#export').href='/api/export?'+filters();","prepareExport(rows);")
js=js.replace("s.available?'● Local AI ready':'Evidence search ready · AI offline'","'● Evidence search · hosted demo'")
js+='\nsetupHostedDemo();\n'
(OUT/'assets/app.js').write_text(js,encoding='utf-8')
shutil.copy2(ROOT/'rocket_landing/web/style.css',OUT/'assets/style.css')
with (OUT/'assets/style.css').open('a',encoding='utf-8') as style:
    style.write('\n@media(max-width:600px){.sidebar nav{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));overflow:visible}.sidebar nav button{gap:5px;padding:9px 5px;white-space:normal}.sidebar nav button span{font-size:10px}}\n')
shutil.copy2(Path(__file__).parent/'demo-api.js',OUT/'assets/demo-api.js')
(OUT/'.nojekyll').touch()
shutil.copy2(Path(__file__).parent/'.gitattributes',OUT/'.gitattributes')
(OUT/'README.md').write_text('# Rocket Landing Lab - interview demo\n\nRead-only browser demonstration of saved simulator evidence. Full source: https://github.com/Oneday-begin/Rocket-Landing-Lab\nTraining and Ollama generation run in the local Python app.\n',encoding='utf-8')
print(f'Exported {len(records)} complete flight histories, saved forecasts, evidence and reports.')
