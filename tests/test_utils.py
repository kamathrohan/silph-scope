import numpy as np
import pytest
from silph_scope.utils import (
    forward_filter,
    backward_sample,
    count_transitions,
    sample_transition_matrix,
    mvn_logpdf_batch,
    hmm_log_emission,
    sample_NIW_s,
    init_regimes,
    relabel_by_mu_v,
    _build_sticky_P,
    _regime_label,
    ess,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_P(K):
    P = _build_sticky_P(K)
    return P


def _make_ll(T, K):
    return RNG.standard_normal((T, K))


# ---------------------------------------------------------------------------
# forward_filter
# ---------------------------------------------------------------------------

class TestForwardFilter:
    def test_shape(self):
        ll = _make_ll(50, 2)
        alpha = forward_filter(ll, _make_P(2))
        assert alpha.shape == (50, 2)

    def test_rows_sum_to_one(self):
        ll = _make_ll(50, 3)
        alpha = forward_filter(ll, _make_P(3))
        np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-10)

    def test_non_negative(self):
        ll = _make_ll(50, 2)
        alpha = forward_filter(ll, _make_P(2))
        assert (alpha >= 0).all()

    def test_certain_regime(self):
        # If one regime has overwhelmingly higher likelihood, filter should concentrate there
        T, K = 30, 2
        ll = np.full((T, K), -1e10)
        ll[:, 0] = 0.0
        alpha = forward_filter(ll, _make_P(K))
        assert (alpha[:, 0] > 0.99).all()


# ---------------------------------------------------------------------------
# backward_sample
# ---------------------------------------------------------------------------

class TestBackwardSample:
    def test_shape(self):
        ll = _make_ll(50, 2)
        alpha = forward_filter(ll, _make_P(2))
        regimes = backward_sample(alpha, _make_P(2))
        assert regimes.shape == (50,)

    def test_valid_regime_indices(self):
        K = 3
        ll = _make_ll(50, K)
        alpha = forward_filter(ll, _make_P(K))
        regimes = backward_sample(alpha, _make_P(K))
        assert regimes.min() >= 0
        assert regimes.max() < K


# ---------------------------------------------------------------------------
# count_transitions
# ---------------------------------------------------------------------------

class TestCountTransitions:
    def test_shape(self):
        regimes = np.array([0, 1, 0, 0, 1])
        n = count_transitions(regimes, K=2)
        assert n.shape == (2, 2)

    def test_counts(self):
        regimes = np.array([0, 1, 0, 1, 0])
        n = count_transitions(regimes, K=2)
        assert n[0, 1] == 2
        assert n[1, 0] == 2
        assert n[0, 0] == 0
        assert n[1, 1] == 0

    def test_total(self):
        regimes = np.array([0, 1, 2, 0, 1])
        n = count_transitions(regimes, K=3)
        assert n.sum() == len(regimes) - 1


# ---------------------------------------------------------------------------
# sample_transition_matrix
# ---------------------------------------------------------------------------

class TestSampleTransitionMatrix:
    def test_rows_sum_to_one(self):
        regimes = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        P, _ = sample_transition_matrix(regimes, K=2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)

    def test_non_negative(self):
        regimes = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        P, _ = sample_transition_matrix(regimes, K=2)
        assert (P >= 0).all()

    def test_shape(self):
        regimes = np.array([0, 1, 2, 0, 1, 2])
        P, n = sample_transition_matrix(regimes, K=3)
        assert P.shape == (3, 3)
        assert n.shape == (3, 3)


# ---------------------------------------------------------------------------
# _build_sticky_P
# ---------------------------------------------------------------------------

class TestBuildStickyP:
    def test_rows_sum_to_one(self):
        for K in (2, 3, 4):
            P = _build_sticky_P(K)
            np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)

    def test_diagonal_is_p_stay(self):
        P = _build_sticky_P(3, p_stay=0.9)
        np.testing.assert_allclose(np.diag(P), 0.9, atol=1e-10)

    def test_non_negative(self):
        assert (_build_sticky_P(4) >= 0).all()


# ---------------------------------------------------------------------------
# mvn_logpdf_batch
# ---------------------------------------------------------------------------

class TestMvnLogpdfBatch:
    def test_shape(self):
        Y = RNG.standard_normal((20, 3))
        mu = np.zeros(3)
        Sigma = np.eye(3)
        ll = mvn_logpdf_batch(Y, mu, Sigma)
        assert ll.shape == (20,)

    def test_standard_normal_at_mean(self):
        # log p(0) for standard univariate normal = -0.5*log(2*pi)
        Y = np.zeros((1, 1))
        mu = np.zeros(1)
        Sigma = np.eye(1)
        ll = mvn_logpdf_batch(Y, mu, Sigma)
        np.testing.assert_allclose(ll[0], -0.5 * np.log(2 * np.pi), atol=1e-10)

    def test_precomputed_matches(self):
        Y = RNG.standard_normal((10, 2))
        mu = RNG.standard_normal(2)
        Sigma = np.eye(2) + 0.3
        Sigma_inv = np.linalg.inv(Sigma)
        _, log_det = np.linalg.slogdet(Sigma)
        ll1 = mvn_logpdf_batch(Y, mu, Sigma)
        ll2 = mvn_logpdf_batch(Y, mu, Sigma, Sigma_inv, log_det)
        np.testing.assert_allclose(ll1, ll2, atol=1e-10)


# ---------------------------------------------------------------------------
# hmm_log_emission
# ---------------------------------------------------------------------------

class TestHmmLogEmission:
    def test_shape(self):
        T, K, d = 30, 3, 2
        Y = RNG.standard_normal((T, d))
        mu_list = [RNG.standard_normal(d) for _ in range(K)]
        Sigma_list = [np.eye(d) for _ in range(K)]
        ll = hmm_log_emission(Y, mu_list, Sigma_list)
        assert ll.shape == (T, K)

    def test_values_match_mvn(self):
        T, d = 10, 2
        Y = RNG.standard_normal((T, d))
        mu = np.zeros(d)
        Sigma = np.eye(d)
        ll = hmm_log_emission(Y, [mu], [Sigma])
        expected = mvn_logpdf_batch(Y, mu, Sigma)
        np.testing.assert_allclose(ll[:, 0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# sample_NIW_s
# ---------------------------------------------------------------------------

class TestSampleNIWs:
    def test_sigma_positive_definite(self):
        Y = RNG.standard_normal((50, 3))
        m0 = np.zeros(3)
        Psi0 = np.eye(3)
        for _ in range(5):
            _, Sigma = sample_NIW_s(Y, m0, kappa0=1.0, Psi0=Psi0, nu0=6)
            eigvals = np.linalg.eigvalsh(Sigma)
            assert (eigvals > 0).all()

    def test_mu_shape(self):
        Y = RNG.standard_normal((30, 3))
        mu, Sigma = sample_NIW_s(Y, np.zeros(3), kappa0=1.0, Psi0=np.eye(3), nu0=6)
        assert mu.shape == (3,)
        assert Sigma.shape == (3, 3)

    def test_sigma_symmetric(self):
        Y = RNG.standard_normal((30, 3))
        _, Sigma = sample_NIW_s(Y, np.zeros(3), kappa0=1.0, Psi0=np.eye(3), nu0=6)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)


# ---------------------------------------------------------------------------
# init_regimes
# ---------------------------------------------------------------------------

class TestInitRegimes:
    def test_regime_indices_valid(self):
        Y = RNG.standard_normal((100, 3))
        for K in (2, 3):
            regimes, mu_list, Sigma_list = init_regimes(Y, K=K)
            assert regimes.min() >= 0
            assert regimes.max() < K
            assert len(mu_list) == K
            assert len(Sigma_list) == K

    def test_all_regimes_populated(self):
        Y = RNG.standard_normal((200, 3))
        regimes, _, _ = init_regimes(Y, K=3)
        assert set(np.unique(regimes)) == {0, 1, 2}

    def test_sigma_positive_definite(self):
        Y = RNG.standard_normal((100, 3))
        _, _, Sigma_list = init_regimes(Y, K=2)
        for S in Sigma_list:
            assert (np.linalg.eigvalsh(S) > 0).all()


# ---------------------------------------------------------------------------
# relabel_by_mu_v
# ---------------------------------------------------------------------------

class TestRelabelByMuV:
    def test_sorted_by_mu_v(self):
        mu_list = [np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0])]
        Sigma_list = [np.eye(3), np.eye(3)]
        regimes = np.array([0, 0, 1, 1])
        mu_new, _, _ = relabel_by_mu_v(mu_list, Sigma_list, regimes, K=2)
        assert mu_new[0][1] < mu_new[1][1]

    def test_already_sorted_unchanged(self):
        mu_list = [np.array([0.0, -1.0, 0.0]), np.array([0.0, 2.0, 0.0])]
        Sigma_list = [np.eye(3), np.eye(3)]
        regimes = np.array([0, 1, 0])
        mu_new, Sigma_new, reg_new = relabel_by_mu_v(mu_list, Sigma_list, regimes, K=2)
        np.testing.assert_array_equal(reg_new, regimes)

    def test_regime_remapping_consistent(self):
        mu_list = [np.array([0.0, 5.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        Sigma_list = [np.eye(3), np.eye(3)]
        regimes = np.array([0, 0, 1])
        _, _, reg_new = relabel_by_mu_v(mu_list, Sigma_list, regimes, K=2)
        # old regime 0 (high mu_v) should become regime 1
        assert reg_new[0] == 1
        assert reg_new[2] == 0


# ---------------------------------------------------------------------------
# _regime_label
# ---------------------------------------------------------------------------

class TestRegimeLabel:
    def test_K2(self):
        assert _regime_label(0, 2) == 'clear_skies'
        assert _regime_label(1, 2) == 'thunderstorm'

    def test_K3(self):
        assert _regime_label(0, 3) == 'clear_skies'
        assert _regime_label(1, 3) == 'sandstorm'
        assert _regime_label(2, 3) == 'thunderstorm'

    def test_K4_fallback(self):
        assert _regime_label(2, 4) == 'type2'


# ---------------------------------------------------------------------------
# ess
# ---------------------------------------------------------------------------

class TestEss:
    def test_iid_close_to_n(self):
        x = RNG.standard_normal(1000)
        assert ess(x) > 500

    def test_highly_autocorrelated_low(self):
        x = np.cumsum(RNG.standard_normal(500))
        assert ess(x) < 100

    def test_short_series(self):
        assert ess(np.array([1.0, 2.0])) <= 2
