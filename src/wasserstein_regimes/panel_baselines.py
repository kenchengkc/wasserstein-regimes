# SPDX-License-Identifier: GPL-3.0-only
"""Training-only panel feature transforms and causal vector-return HMM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from .sliced import _joint
from .transport import _positive_int


def panel_features(samples, kind):
    x = _joint(samples)
    if kind == 'marginal':
        return np.sort(x, axis=1).reshape(len(x), -1) / np.sqrt(x.shape[1]*x.shape[2])
    centered = x-x.mean(axis=1,keepdims=True)
    cov = np.einsum('nld,nle->nde',centered,centered)/x.shape[1]
    if kind == 'covariance':
        i,j = np.triu_indices(x.shape[2])
        return cov[:,i,j]
    if kind != 'correlation' or x.shape[2] < 2:
        raise ValueError('unknown feature kind or insufficient assets')
    vol = np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),0))
    denom = vol[:,:,None]*vol[:,None,:]
    corr = np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
    i,j = np.triu_indices(x.shape[2],1)
    return np.clip(corr[:,i,j],-1,1)


def _read(path, keys, kind):
    try:
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files) != set(keys)|{'metadata'}:
                raise ValueError('invalid baseline keys')
            metadata = json.loads(str(saved['metadata'].item()))
            if metadata['version'] != 1 or metadata['type'] != kind:
                raise ValueError('invalid baseline version')
            arrays = {name:saved[name].copy() for name in keys}
        if any(np.iscomplexobj(a) or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all()
               for a in arrays.values()):
            raise ValueError('invalid baseline arrays')
        return metadata, arrays
    except (OSError,ValueError,TypeError,KeyError) as error:
        raise ValueError('invalid baseline archive') from error


def _save(path,metadata,**arrays):
    with Path(path).open('wb') as handle:
        np.savez_compressed(handle,metadata=json.dumps(dict(version=1,**metadata)),**arrays)


class PanelKMeans:
    def __init__(self,kind,n_clusters,*,scales=None,n_init=20,seed=42):
        if kind not in ('marginal','covariance','correlation'):
            raise ValueError('unknown feature kind')
        self.kind=kind
        self.n_clusters=_positive_int(n_clusters,'n_clusters')
        self.n_init=_positive_int(n_init,'n_init')
        self.seed=seed
        self.scales=None if scales is None else np.asarray(scales,dtype=float).copy()

    def fit(self,samples):
        x=_joint(samples)
        scales=np.ones(x.shape[2]) if self.scales is None else self.scales.copy()
        if scales.shape!=(x.shape[2],) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError('invalid asset scales')
        f=panel_features(x/scales,self.kind)
        mean=np.zeros(f.shape[1]) if self.kind=='marginal' else f.mean(axis=0)
        std=np.ones(f.shape[1]) if self.kind=='marginal' else f.std(axis=0)
        std=np.where(std>1e-12,std,1.)
        km=KMeans(n_clusters=self.n_clusters,n_init=self.n_init,random_state=self.seed).fit((f-mean)/std)
        self.scales_,self.feature_mean_,self.feature_std_=scales,mean,std
        self.centers_,self.input_shape_=km.cluster_centers_.copy(),x.shape[1:]
        self.training_occupied_=int(len(np.unique(km.labels_)))
        return self

    def predict(self,samples):
        if not hasattr(self,'centers_'):
            raise ValueError('model must be fitted')
        x=_joint(samples)
        if x.shape[1:]!=self.input_shape_:
            raise ValueError('incompatible panel shape')
        f=(panel_features(x/self.scales_,self.kind)-self.feature_mean_)/self.feature_std_
        costs=np.column_stack([np.sum((f-c)**2,axis=1) for c in self.centers_])
        return costs.argmin(axis=1)

    def save(self,path):
        _save(path,dict(type='panel_kmeans',kind=self.kind,n_clusters=self.n_clusters,
                        input_shape=list(self.input_shape_),training_occupied=self.training_occupied_),
              scales=self.scales_,mean=self.feature_mean_,std=self.feature_std_,centers=self.centers_)

    @classmethod
    def load(cls,path):
        m,a=_read(path,['scales','mean','std','centers'],'panel_kmeans')
        model=cls(m['kind'],m['n_clusters'],scales=a['scales'])
        shape=tuple(m['input_shape'])
        if len(shape)!=2 or any(not isinstance(v,int) or v<1 for v in shape):
            raise ValueError('invalid input shape')
        width=shape[0]*shape[1] if model.kind=='marginal' else shape[1]*(shape[1]+(1 if model.kind=='covariance' else -1))//2
        if (a['scales'].shape!=(shape[1],) or np.any(a['scales']<=0)
                or a['centers'].shape!=(model.n_clusters,width)
                or a['mean'].shape!=(width,) or a['std'].shape!=(width,) or np.any(a['std']<=0)):
            raise ValueError('invalid feature state')
        model.scales_,model.feature_mean_,model.feature_std_=a['scales'],a['mean'],a['std']
        model.centers_,model.input_shape_=a['centers'],shape
        model.training_occupied_=m['training_occupied']
        return model


def _vectors(values):
    if np.iscomplexobj(values):
        raise ValueError('returns must be real')
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('returns must be a finite nonempty matrix')
    return x


class PanelGaussianHMM:
    """Full-covariance emissions on daily vectors; forward filtering only."""
    def __init__(self,n_clusters,*,n_init=3,max_iter=200,seed=42):
        self.n_clusters=_positive_int(n_clusters,'n_clusters')
        self.n_init=_positive_int(n_init,'n_init')
        self.max_iter=_positive_int(max_iter,'max_iter')
        self.seed=seed

    def fit(self,returns):
        from hmmlearn.hmm import GaussianHMM
        x=_vectors(returns)
        d=x.shape[1]
        free=self.n_clusters**2-1+self.n_clusters*(d+d*(d+1)//2)
        if len(x)<free:
            raise ValueError('insufficient HMM training observations')
        mean=x.mean(axis=0)
        std=x.std(axis=0)
        if np.any(std<=1e-12):
            raise ValueError('constant HMM training asset')
        z=(x-mean)/std
        best=None
        diagnostics=[]
        for seed in np.random.default_rng(self.seed).integers(0,2**31-1,size=self.n_init):
            row=dict(seed=int(seed))
            try:
                candidate=GaussianHMM(n_components=self.n_clusters,covariance_type='full',
                                      min_covar=1e-6,n_iter=self.max_iter,random_state=int(seed)).fit(z)
                score=float(candidate.score(z))
                valid=(np.isfinite(score) and np.isfinite(candidate.means_).all()
                       and np.isfinite(candidate.covars_).all()
                       and all(np.linalg.eigvalsh(c).min()>0 for c in candidate.covars_))
                history=list(candidate.monitor_.history)
                # hmmlearn also calls reaching max_iter "converged"; distinguish it.
                converged=(len(history)>1 and 0<=history[-1]-history[-2]<candidate.tol)
                row.update(loglikelihood=score,converged=converged,n_iter=int(candidate.monitor_.iter),
                           reached_iteration_limit=candidate.monitor_.iter>=self.max_iter)
                if valid and (best is None or score>best[0]):
                    best=(score,candidate)
            except (ValueError,np.linalg.LinAlgError) as exc:
                row['error']=str(exc)
            diagnostics.append(row)
        if best is None:
            raise ValueError('all HMM fits failed')
        candidate=best[1]
        self.train_mean_,self.train_std_=mean,std
        self.means_,self.covars_=candidate.means_.copy(),candidate.covars_.copy()
        self.startprob_=np.maximum(candidate.startprob_,1e-12)
        self.startprob_/=self.startprob_.sum()
        self.transmat_=np.maximum(candidate.transmat_,1e-12)
        self.transmat_/=self.transmat_.sum(axis=1,keepdims=True)
        self.diagnostics_=diagnostics
        self.selected_loglikelihood_=best[0]
        return self

    def filter_proba(self,returns):
        if not hasattr(self,'means_'):
            raise ValueError('model must be fitted')
        x=_vectors(returns)
        if x.shape[1]!=len(self.train_mean_):
            raise ValueError('incompatible HMM asset dimension')
        z=(x-self.train_mean_)/self.train_std_
        emissions=np.column_stack([multivariate_normal.logpdf(z,mean=mean,cov=cov)
                                   for mean,cov in zip(self.means_,self.covars_)])
        log_trans=np.log(self.transmat_)
        out=np.empty_like(emissions)
        for t,row in enumerate(emissions):
            prior=np.log(self.startprob_) if t==0 else logsumexp(alpha[:,None]+log_trans,axis=0)
            alpha=prior+row
            alpha-=logsumexp(alpha)
            out[t]=np.exp(alpha)
        return out

    def predict(self,returns):
        return self.filter_proba(returns).argmax(axis=1)

    def save(self,path):
        _save(path,dict(type='panel_hmm',n_clusters=self.n_clusters,diagnostics=self.diagnostics_,
                        selected_loglikelihood=self.selected_loglikelihood_),
              mean=self.train_mean_,std=self.train_std_,means=self.means_,covars=self.covars_,
              start=self.startprob_,trans=self.transmat_)

    @classmethod
    def load(cls,path):
        m,a=_read(path,['mean','std','means','covars','start','trans'],'panel_hmm')
        model=cls(m['n_clusters'])
        k=model.n_clusters
        d=a['mean'].size
        if (a['mean'].shape!=(d,) or a['std'].shape!=(d,) or np.any(a['std']<=0)
                or a['means'].shape!=(k,d) or a['covars'].shape!=(k,d,d)
                or a['start'].shape!=(k,) or a['trans'].shape!=(k,k)
                or np.any(a['start']<=0) or np.any(a['trans']<=0)
                or not np.isclose(a['start'].sum(),1)
                or not np.allclose(a['trans'].sum(axis=1),1)
                or any(not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min()<=0 for c in a['covars'])):
            raise ValueError('invalid HMM state')
        model.train_mean_,model.train_std_=a['mean'],a['std']
        model.means_,model.covars_=a['means'],a['covars']
        model.startprob_,model.transmat_=a['start'],a['trans']
        model.diagnostics_=m['diagnostics']
        model.selected_loglikelihood_=m['selected_loglikelihood']
        return model
