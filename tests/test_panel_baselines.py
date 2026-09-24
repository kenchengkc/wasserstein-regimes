# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest

from wasserstein_regimes.panel_baselines import PanelKMeans, PanelGaussianHMM, panel_features


def test_feature_geometries_preserve_marginals_and_joint_covariance():
    a = np.array([[-1.,-1.],[1.,1.]])
    b = np.array([[-1.,1.],[1.,-1.]])
    x = np.stack([a,b])
    assert np.array_equal(panel_features(x,'marginal')[0],panel_features(x,'marginal')[1])
    np.testing.assert_allclose(panel_features(x,'covariance'), [[1,1,1],[1,-1,1]])
    np.testing.assert_allclose(panel_features(x,'correlation'), [[1],[-1]])


@pytest.mark.parametrize('kind', ['marginal','covariance','correlation'])
def test_feature_fit_freezes_state_and_roundtrips(kind,tmp_path):
    x = np.random.default_rng(8).normal(size=(50,9,3))
    model = PanelKMeans(kind,2,n_init=2,scales=[1,2,3]).fit(x)
    before = model.predict(x)
    mean = model.feature_mean_.copy()
    model.predict(x*100)
    np.testing.assert_array_equal(model.feature_mean_,mean)
    model.save(tmp_path/'model.npz')
    restored = PanelKMeans.load(tmp_path/'model.npz')
    np.testing.assert_array_equal(before,restored.predict(x))


def test_hmm_prefix_invariance_and_reload(tmp_path):
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-1,.2,(120,2)),rng.normal(1,.2,(120,2))]
    model = PanelGaussianHMM(2,n_init=2,max_iter=50).fit(x)
    prefix = model.filter_proba(x[:180])
    all_probs = model.filter_proba(np.r_[x[:180],np.full((30,2),100)])
    np.testing.assert_allclose(prefix,all_probs[:180])
    np.testing.assert_allclose(prefix.sum(axis=1),1)
    model.save(tmp_path/'hmm.npz')
    restored = PanelGaussianHMM.load(tmp_path/'hmm.npz')
    np.testing.assert_allclose(restored.filter_proba(x),model.filter_proba(x))
    assert model.diagnostics_


def test_invalid_inputs_fail():
    x = np.ones((10,5,2))
    with pytest.raises(ValueError):
        panel_features(x,'unknown')
    with pytest.raises(ValueError):
        PanelKMeans('marginal',2,scales=[0,1]).fit(x)
    with pytest.raises(ValueError):
        PanelGaussianHMM(2).fit(np.full((100,2),np.nan))
