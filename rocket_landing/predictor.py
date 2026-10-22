"""Predict outcomes at a fixed 150-step horizon without future-state leakage."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .telemetry import DATA, DB, flight_rows, read_flight

HORIZON = 150
LABELS = ['failed', 'landed', 'timeout']
FEATURES = [f'{stat}_{name}' for stat in ['last','mean','std','min','max']
            for name in ['x','y','vx','vy','angle','spin']] + ['main_fraction','side_fraction']


def features(samples):
    # Input step 0 is reset. Only observations available through step 150 enter this vector.
    if len(samples) <= HORIZON: raise ValueError('Flight ended before the three-second observation window')
    values = np.asarray(samples[:HORIZON+1], dtype=float)
    obs = values[:,1:7]
    actions = values[1:,9]
    return np.concatenate([obs[-1],obs.mean(0),obs.std(0),obs.min(0),obs.max(0),
                           [(actions==2).mean(), np.isin(actions,[1,3]).mean()]])


def metrics(model, x, y):
    predictions=model.predict(x)
    probabilities=model.predict_proba(x)
    success_index=list(model.classes_).index('landed')
    return {'accuracy':float(accuracy_score(y,predictions)),
            'macro_f1':float(f1_score(y,predictions,labels=LABELS,average='macro',zero_division=0)),
            'landing_brier':float(brier_score_loss((y=='landed').astype(int),probabilities[:,success_index])),
            'confusion_matrix':confusion_matrix(y,predictions,labels=LABELS).tolist(),
            'classification':classification_report(y,predictions,labels=LABELS,output_dict=True,zero_division=0)}


def train_predictor(database=DB, output=DATA):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=flight_rows(database)
    eligible=[r for r in rows if r['steps']>HORIZON]
    if len(eligible)<60: raise ValueError('Collect at least 60 eligible flights first')
    x=np.asarray([features(read_flight(r['id'],database)['samples']) for r in eligible])
    y=np.asarray([r['outcome'] for r in eligible]);groups=np.asarray([r['seed'] for r in eligible])
    train_val,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(x,y,groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=43).split(x[train_val],y[train_val],groups[train_val]))
    train,validation=train_val[tr],train_val[va]
    assert not (set(groups[train]) & set(groups[test]) or set(groups[validation]) & set(groups[test]) or set(groups[train]) & set(groups[validation]))
    if set(y[train]) != set(LABELS): raise ValueError('Training partition needs all three outcomes')
    candidates={'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=42)),
                'random_forest':RandomForestClassifier(n_estimators=180,max_depth=9,min_samples_leaf=4,random_state=42,n_jobs=1)}
    validation_results={}
    for name,model in candidates.items():
        model.fit(x[train],y[train]);validation_results[name]=metrics(model,x[validation],y[validation])
    winner=max(validation_results,key=lambda n:validation_results[n]['macro_f1'])
    # Keep the selected trained model unchanged; test data never fits model or preprocessing.
    model=candidates[winner]
    test_metrics=metrics(model,x[test],y[test])
    probs=model.predict_proba(x[test])[:,list(model.classes_).index('landed')]
    reliability=[]
    for low in np.arange(0,1,.2):
        mask=(probs>=low)&(probs<(low+.2) if low<.8 else probs<=1)
        if mask.any(): reliability.append({'predicted':float(probs[mask].mean()),'observed':float((y[test][mask]=='landed').mean()),'count':int(mask.sum())})
    split_indices={'train':train,'validation':validation,'test':test}
    assignments={eligible[i]['id']:name for name,indices in split_indices.items() for i in indices}
    fingerprint=hashlib.sha256('\n'.join(sorted(r['id'] for r in eligible)).encode()).hexdigest()
    joblib.dump({'model':model,'features':FEATURES,'horizon':HORIZON,'assignments':assignments},output/'predictor.joblib')
    result={'selected':winner,'horizon_steps':HORIZON,'horizon_seconds':3,'features':FEATURES,'labels':LABELS,
            'total_flights':len(rows),'eligible_flights':len(eligible),'early_endings_excluded':len(rows)-len(eligible),
            'validation_comparison':validation_results,'test':test_metrics,'reliability':reliability,
            'splits':{name:{'flights':len(indices),'seeds':sorted(set(groups[indices].tolist()))} for name,indices in split_indices.items()},
            'library_fingerprint':fingerprint,'model_sha256':hashlib.sha256((output/'predictor.joblib').read_bytes()).hexdigest(),
            'limitation':'Known policies on unseen seed groups. Probabilities are estimates; reliability is measured, not guaranteed. Flights ending before or at 150 steps are excluded.'}
    (output/'predictor-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'selected':winner,'test':test_metrics,'eligible':len(eligible)},indent=2),flush=True)
    return result


def predict_flight(flight, output=DATA):
    if flight['steps']<=HORIZON: return {'available':False,'reason':'This flight already ended by three seconds. No early forecast is shown.'}
    path=Path(output)/'predictor.joblib'
    if not path.exists(): return {'available':False,'reason':'Train the predictor after collecting flights.'}
    report=json.loads((Path(output)/'predictor-report.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('Predictor changed after evaluation; retrain before using it')
    saved=joblib.load(path)
    vector=features(flight['samples']).reshape(1,-1)
    probs=saved['model'].predict_proba(vector)[0]
    return {'available':True,'at_step':HORIZON,'split':saved['assignments'].get(flight['id'],'new flight'),
            'probabilities':dict(zip(saved['model'].classes_,map(float,probs))),
            'prediction':str(saved['model'].predict(vector)[0]),'outcome':flight['outcome']}


if __name__=='__main__':train_predictor()
