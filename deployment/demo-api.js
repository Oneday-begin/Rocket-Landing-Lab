const demoCache=new Map();
async function demoJson(path){if(!demoCache.has(path))demoCache.set(path,fetch(path).then(r=>{if(!r.ok)throw Error('Could not load demo evidence. Refresh the page.');return r.json()}));return demoCache.get(path)}
function demoFilter(data,p){return data.filter(r=>['model','scenario','outcome'].every(k=>!p.get(k)||r[k]===p.get(k)))}
function interval(k,n){const z=1.95996398454,p=k/n,d=1+z*z/n,mid=(p+z*z/(2*n))/d,delta=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [mid-delta,mid+delta]}
async function api(path,options={}){
 const url=new URL(path,'https://demo.local'),p=url.searchParams;
 if(url.pathname==='/api/jobs')return {state:'idle',lines:[]};
 if(url.pathname==='/api/assistant/status')return {available:false};
 if(url.pathname==='/api/predictor')return demoJson('data/predictor.json');
 if(url.pathname==='/api/training')return demoJson('data/training.json');
 if(url.pathname==='/api/ask')return demoAnswer(JSON.parse(options.body).question);
 if(url.pathname.startsWith('/api/flights/'))return demoJson('data/flights/'+url.pathname.split('/').pop()+'.json');
 const data=demoFilter(await demoJson('data/flights.json'),p);
 if(url.pathname==='/api/flights')return data;
 if(url.pathname==='/api/summary'){
  const models=[...new Set(data.map(r=>r.model))].map(model=>{const rs=data.filter(r=>r.model===model),n=rs.length,k=rs.filter(r=>r.outcome==='landed').length;return {model,episodes:n,landed:k/n,ci:interval(k,n),score:rs.reduce((s,r)=>s+r.score,0)/n,fuel:rs.reduce((s,r)=>s+r.fuel,0)/n,timeout:rs.filter(r=>r.outcome==='timeout').length/n}});
  return {count:data.length,models,landed:data.filter(r=>r.outcome==='landed').length,mean_score:data.length?data.reduce((s,r)=>s+r.score,0)/data.length:0,mean_seconds:data.length?data.reduce((s,r)=>s+r.steps/50,0)/data.length:0,seed_range:data.length?[Math.min(...data.map(r=>r.seed)),Math.max(...data.map(r=>r.seed))]:[]};
 }
 throw Error('This action is available in the full local lab.');
}
let exportUrl;
function prepareExport(data){const keys=['id','model','scenario','seed','model_sha256','outcome','score','steps','fuel','hover'],quote=v=>'"'+String(v).replaceAll('"','""')+'"';const csv=[keys.join(','),...data.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\r\n');if(exportUrl)URL.revokeObjectURL(exportUrl);exportUrl=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const a=document.querySelector('#export');a.href=exportUrl;a.download='rocket-flights.csv'}
function setupHostedDemo(){for(const [id,label] of [['collect','Collection runs in local app'],['fit-predictor','Training runs in local app'],['train-broader','Training runs in local app']]){const b=document.getElementById(id);b.disabled=true;b.textContent=label;b.title='This free hosted demo shows recorded evidence. Use Open-Web-Lab.cmd on your laptop for new experiments.'}document.querySelector('#training-steps').disabled=true}
async function demoAnswer(question){
 const q=question.toLowerCase(),weather=['gusty','breezy','calm'].find(w=>q.includes(w));
 if(['best','landing rate','success rate','how many flights','average score'].some(w=>q.includes(w))){
  const data=(await demoJson('data/flights.json')).filter(r=>!weather||r.scenario===weather);let answer;
  if(q.includes('how many flights'))answer=`There are ${data.length} recorded flights${weather?' in '+weather+' weather':''}.`;
  else{const groups=[...new Set(data.map(r=>r.model))].map(model=>{const rs=data.filter(r=>r.model===model);return {model,n:rs.length,rate:rs.filter(r=>r.outcome==='landed').length/rs.length,score:rs.reduce((s,r)=>s+r.score,0)/rs.length}}).sort((a,b)=>q.includes('average score')?b.score-a.score:b.rate-a.rate);answer=`Saved telemetry${weather?' in '+weather+' weather':''}:\n`+groups.map(r=>`${r.model}: ${(r.rate*100).toFixed(1)}% landed, mean original score ${r.score.toFixed(1)}, ${r.n} flights.`).join('\n')+'\nThese results describe these saved policies and seeds, not universal superiority.'}
  return {answer,mode:'computed data',sources:[{id:'DATA',title:'Recorded flight library',text:'720 flights; original simulator rewards; seed numbers 600000-600059; deterministic saved policies.',path:'data/flights.json'}]};
 }
 const docs=await demoJson('data/evidence.json'),terms=[...new Set(q.match(/[a-z0-9]+/g)||[])].filter(t=>t.length>2&&!['the','why','how','did','does','what','this','that','with','and','for','can','are'].includes(t));
 const scored=docs.map(d=>{const text=d.text.toLowerCase();return {...d,score:terms.reduce((n,t)=>n+(text.includes(t)?1:0),0)}}).filter(d=>d.score>=Math.min(2,Math.max(1,terms.length))).sort((a,b)=>b.score-a.score).slice(0,3);
 if(!scored.length)return {answer:'No supporting project evidence found. Try asking about PPO, reward shaping, telemetry, the predictor or weather training.',mode:'no evidence',sources:[]};
 return {answer:'Relevant project excerpts are shown below. This phone demo searches evidence; local language generation runs in the full desktop lab.',mode:'retrieved excerpts',sources:scored};
}
