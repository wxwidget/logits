"""Reproducible dense/CSR solver comparison; emits JSON plus CSV traces."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from logitlab import LogisticRegression


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=10000)
    p.add_argument('--cols', type=int, default=100)
    p.add_argument('--density', type=float, default=1., help='Use CSR when below 1')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--learning-rate', type=float, default=.05)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--l2', type=float, default=.01)
    p.add_argument('--tolerance', type=float, default=1e-6)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--solvers', nargs='+', default=['gd','sgd','adam','lbfgs','newton-cg'])
    p.add_argument('--output', type=Path, default=Path('build/benchmark'))
    a=p.parse_args()
    if a.rows<10 or a.cols<1 or not 0<a.density<=1 or a.epochs<1:
        p.error('rows >= 10, cols >= 1, 0 < density <= 1, epochs >= 1 required')
    a.output.mkdir(parents=True,exist_ok=True)
    results=[];traces=[]
    for seed in a.seeds:
        rng=np.random.default_rng(seed)
        if a.density<1:
            X=sparse.random(a.rows,a.cols,density=a.density,format='csr',random_state=rng,
                            data_rvs=lambda n:rng.normal(size=n))
        else:
            X=rng.normal(size=(a.rows,a.cols))
        truth=rng.normal(size=a.cols)
        z=np.asarray(X@truth).ravel()/np.sqrt(max(1,a.cols*a.density))
        y=(rng.random(a.rows)<1/(1+np.exp(-z))).astype(float)
        cut=int(a.rows*.8)
        train,test=X[:cut],X[cut:];yt,yv=y[:cut],y[cut:]
        # Training-only scaling without centering preserves sparse structure.
        if sparse.issparse(train):
            scale=np.sqrt(np.asarray(train.power(2).mean(axis=0)).ravel())
            scale[scale==0]=1
            train=train.multiply(1/scale).tocsr();test=test.multiply(1/scale).tocsr()
        else:
            scale=np.sqrt(np.mean(train**2,axis=0));scale[scale==0]=1
            train=train/scale;test=test/scale
        for solver in a.solvers:
            start=time.perf_counter()
            model=LogisticRegression(solver=solver,max_epochs=a.epochs,l2=a.l2,
                                     learning_rate=a.learning_rate,batch_size=a.batch_size,
                                     tolerance=a.tolerance,seed=seed).fit(train,yt)
            elapsed=time.perf_counter()-start
            prob=model.predict_proba(test)[:,1];clipped=np.clip(prob,1e-15,1-1e-15)
            row=dict(seed=seed,solver=solver,fit_seconds=elapsed,
                     train_objective=model.history_[-1]['loss'],
                     gradient_norm=model.history_[-1]['gradient_norm'],
                     validation_log_loss=float(-np.mean(yv*np.log(clipped)+(1-yv)*np.log1p(-clipped))),
                     validation_accuracy=float(np.mean((prob>=.5)==yv)),
                     epochs=model.history_[-1]['epoch'],status=model.status_,
                     train_rows=cut,cols=a.cols,train_nnz=int(train.nnz) if sparse.issparse(train) else int(train.size))
            results.append(row)
            traces.extend(dict(seed=seed,solver=solver,**h) for h in model.history_)
            print(json.dumps(row),flush=True)
    report=dict(config=vars(a)|{'output':str(a.output)},
                machine=dict(platform=platform.platform(),processor=platform.processor(),
                             python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                results=results)
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'traces.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','solver','epoch','loss','gradient_norm','seconds'])
        writer.writeheader();writer.writerows(traces)

if __name__=='__main__':main()
