import numpy as np
import numba
from scipy.stats import gaussian_kde
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import brentq
import seaborn as sns
import matplotlib.pyplot as plt


@numba.njit
def trmean(x, tr):
    # calculated truncated mean
    lower = x < np.quantile(x, tr)
    upper = x > np.quantile(x, 1.0 - tr)
    return x[~(lower | upper)].mean()


@numba.njit
def logmean(x, tr):
    return np.exp(np.log(x).mean())


@numba.njit
def bootstrap_lr(lr, n=10000, trunc=0.25):
    ''' gets distribution of the likelihood ratio
    by bootstrapping. If multiple ratios (from multiple samples)
    are provided combines them multiplicatively
    lr - list of numpy arrays of likelihood ratios
    '''
    res = np.ones(n)
    for a in lr:
        res_a = np.zeros(n)
        if trunc > 0.49:
            for i in range(n):
                sample_a = np.median(np.random.choice(
                    a, size=a.size, replace=True))
                res_a[i] = sample_a

        else:
            for i in range(n):
                sample_a = trmean(np.random.choice(
                    a, size=a.size, replace=True), tr=trunc)
                res_a[i] = sample_a
        res *= res_a
    return res


class Decider(object):
    """Decider - a class that performs Bayesian-optimal
    decision for the given set of spectra"""

    def __init__(self, nnmodel, x_train, y_train,
                 prior_positive=0.5,
                 lmbd_fp=1,
                 lmbd_fn=1,
                 epsilon=0.1,
                 n_samples_reference=10,
                 n_samples_prediction=50):
        ''' 
        nnmodel - tf.keras model that maps input spectra into 1D embeddings
        prior_positive - positive prior probability
        lmbd_fp - loss of false positive error
        lmbd_fn - loss of false negative error
        epsilon - loss of "do not know" decision
        n_samples_reference - how many times reference distributions are sampled
        n_samples_prediction - how many times distributions are sampled during prediction
        '''
        super(Decider, self).__init__()

        self.nnmodel = nnmodel
        self.prior_positive = prior_positive
        self.prior_negative = 1 - self.prior_positive
        self.lmbd_fn = lmbd_fn
        self.lmbd_fp = lmbd_fp
        self.epsilon = epsilon

        self.theta_positive = (self.prior_negative / self.prior_positive) * \
            (self.lmbd_fp - self.epsilon) / self.epsilon
        self.theta_negative = (self.prior_positive / self.prior_negative) * \
            (self.lmbd_fn - self.epsilon) / self.epsilon

        self.params = {}

        self.params['n_samples_reference'] = n_samples_reference
        self.params['n_samples_prediction'] = n_samples_prediction
        self.params['delta_risk_max'] = 0.01
        self.params['n_bootstrap'] = 1000
        self.params['mral_interval_a'] = 0.05
        self.params['mral_interval_b'] = 0.95
        self.params['truncation'] = 0.25
        self.params['convergence_threshold'] = 0.002
        self.params['convergence_n_repeat'] = 15
        self.params['convergence_n_last'] = 3

        self.emb_positive = [self.nnmodel.predict(x_train)[y_train == 0]
                             for i in np.arange(self.params['n_samples_reference'])]
        self.emb_negative = [self.nnmodel.predict(x_train)[y_train == 1]
                             for i in np.arange(self.params['n_samples_reference'])]

        self.emb_positive = np.array(self.emb_positive).flatten()
        self.emb_negative = np.array(self.emb_negative).flatten()

        self.positive_med = np.median(self.emb_positive)
        self.negative_med = np.median(self.emb_negative)
        self.positive_kde = gaussian_kde(self.emb_positive)
        self.negative_kde = gaussian_kde(self.emb_negative)

        self.endgrid = max(np.percentile(self.emb_positive, 99.99),
                           np.percentile(self.emb_negative, 99.99))
        self.startgrid = min(np.percentile(
            self.emb_positive, 0.01), np.percentile(self.emb_negative, 0.01))
        self.fullgrid = np.linspace(self.startgrid, self.endgrid, 1000)

        self.positive_dens = self.positive_kde(self.fullgrid)
        self.negative_dens = self.negative_kde(self.fullgrid)

        self.iqr_positive = np.abs(np.quantile(
            self.emb_positive, 0.75) - np.quantile(self.emb_positive, 0.25))
        self.iqr_negative = np.abs(np.quantile(
            self.emb_negative, 0.75) - np.quantile(self.emb_negative, 0.25))

        self.spline_positive = interpolate.InterpolatedUnivariateSpline(
            self.fullgrid, self.positive_kde(self.fullgrid))
        self.spline_negative = interpolate.InterpolatedUnivariateSpline(
            self.fullgrid, self.negative_kde(self.fullgrid))

        self.interval_a = min(
            [self.positive_med - self.iqr_positive, self.negative_med - self.iqr_negative])
        self.interval_b = max(
            [self.positive_med + self.iqr_positive, self.negative_med + self.iqr_negative])

        self.last_mral_positive = None
        self.last_mral_negative = None
        self.last_emb = None

    def update_params(self, prior_positive=None, lmbd_fp=None, lmbd_fn=None, epsilon=None):
        '''
        updates decider parameters like losses and priors
        '''
        if prior_positive is not None:
            self.prior_positive = prior_positive
            self.prior_negative = 1 - self.prior_positive
        if lmbd_fn is not None:
            self.lmbd_fn = lmbd_fn
        if lmbd_fp is not None:
            self.lmbd_fp = lmbd_fp
        if epsilon is not None:
            self.epsilon = epsilon
        self.theta_positive = (self.prior_negative / self.prior_positive) * \
            (self.lmbd_fp - self.epsilon) / self.epsilon
        self.theta_negative = (self.prior_positive / self.prior_negative) * \
            (self.lmbd_fn - self.epsilon) / self.epsilon

    def risk_positive(self, lr):
        # risk function of the decision +
        return self.lmbd_fp * (1 / (1 + lr * (self.prior_positive / self.prior_negative)))

    def risk_dontknow(self, lr):
        # risk function of the decision ?
        return np.ones_like(lr) * self.epsilon

    def risk_negative(self, lr):
        # risk function of the decision -
        return self.lmbd_fn * (1 / (1 + lr * (self.prior_negative / self.prior_positive)))

    def embed(self, data, mult=50):
        # maps spectra to the embeddings with nnmodel
        embeddings = []
        for i in range(mult):
            emb_t = self.nnmodel(data, training=True)
            embeddings.append(emb_t)

        return np.hstack(embeddings)

    def remove_bad_spectra(self, sample):
        # remove outliers that are further from origin than reference
        # distributions plus their IQR
        crit_a = np.median(sample, axis=1) > max(
            [self.positive_med + self.iqr_positive, self.negative_med + self.iqr_negative])
        crit_b = np.median(sample, axis=1) < min(
            [self.positive_med - self.iqr_positive, self.negative_med - self.iqr_negative])
        sample = sample[~(crit_a | crit_b)]
        print(f"{(crit_a | crit_b).sum()}/{sample.shape[0]+(crit_a | crit_b).sum()} bad samples found")
        return sample

    def likelihood_ratio(self, data):
        '''
        gets likelihood ratios from the sampled embeddings
        by calculating average likelihoods with reference distributions

        data: list of numpy arrays of spectra embeddings
        '''
        lr_positive = []
        lr_negative = []
        for arr in data:
            kdes = np.apply_along_axis(gaussian_kde, axis=1, arr=arr)
            likelihoods_positive = np.array(
                [np.trapz(k(self.fullgrid) * self.positive_dens, x=self.fullgrid) for k in kdes])
            likelihoods_negative = np.array(
                [np.trapz(k(self.fullgrid) * self.negative_dens, x=self.fullgrid) for k in kdes])

            lr_positive.append(likelihoods_positive / likelihoods_negative)
            lr_negative.append(likelihoods_negative / likelihoods_positive)
        return tuple(lr_positive), tuple(lr_negative)

    def decide(self, lr_positive, lr_negative):
        '''
        main decision routine
        lr_positive - list of numpy arrays of likelihood ratios +/-
        lr_negative - list of numpy arrays of likelihood rations -/+
        '''

        # get MRAL for positive and negative cases
        mral_positive = bootstrap_lr(
            lr_positive, trunc=self.params['truncation'], n=self.params['n_bootstrap'])
        mral_negative = bootstrap_lr(
            lr_negative, trunc=self.params['truncation'], n=self.params['n_bootstrap'])

        self.last_mral_positive = mral_positive
        self.last_mral_negative = mral_negative

        # print some debug information
        print("MRAL+: [{:.3f}, {:.3f}]".format(np.quantile(mral_positive, self.params["mral_interval_a"]),
                                               np.quantile(mral_positive, self.params["mral_interval_b"])))
        print("MRAL-: [{:.3f}, {:.3f}]".format(np.quantile(mral_negative, self.params["mral_interval_a"]),
                                               np.quantile(mral_negative, self.params["mral_interval_b"])))
        print("th+: {}, th-: {}".format(self.theta_positive, self.theta_negative))

        # 1. if whole distribution is far enough from threshold
        # decision is simple
        if np.quantile(mral_positive, self.params['mral_interval_a']) > self.theta_positive:
            return "POSITIVE"

        if np.quantile(mral_negative, self.params['mral_interval_a']) > self.theta_negative:
            return "NEGATIVE"

        # 2. if theta is inside mral_positive, need to analyze it in more details
        if (self.theta_positive < np.quantile(mral_positive, self.params['mral_interval_b']) and
                self.theta_positive > np.quantile(mral_positive, self.params['mral_interval_a'])):
            return self.analyze_deltarisk(lr_positive,
                                          self.theta_positive,
                                          self.risk_positive,
                                          self.risk_dontknow,
                                          "POSITIVE")
        # similarly if theta is inside mral_negative
        if (self.theta_negative < np.quantile(mral_negative, self.params['mral_interval_b']) and
                self.theta_negative > np.quantile(mral_negative, self.params['mral_interval_a'])):
            return self.analyze_deltarisk(lr_negative,
                                          self.theta_negative,
                                          self.risk_negative,
                                          self.risk_dontknow,
                                          "NEGATIVE")

        # 3. else we purely dont now
        return "DON'T KNOW"

    def analyze_deltarisk(self, lr, theta, risk, risk_dontknow, label=None):
        '''
        decides whether possible risk reduction after collection of additional
        samples significant

        lr - list of numpy arrays of likelihood ratios
        theta - current decision threshold
        risk - risk function of the decision (+ or -)
        risk_dontknow - risk_function of the "don't know" decision
        '''

        # if possible risk reduction is not significant
        # return current optimal decision
        rel_delta_risk, dontknow = self.get_rel_delta_risk(
            lr, theta, risk, risk_dontknow)
        if rel_delta_risk < self.params['delta_risk_max']:
            if dontknow:
                return "DON'T KNOW"
            else:
                return label
        else:
            # if number of spectra seems enough, i.e. delta risk has
            # converged, than nothing else can be said
            conv = self.get_delta_convergence(lr, theta, risk, risk_dontknow)
            if conv:
                return f"{label} \n WARNING: result is on boundary, another sample is recommended"
            # if conergence not yet reached, collectio of additional spectra
            # may help with decision
            else:
                return f"NEED MORE SPECTRA, so far looks like {label}"

    def get_rel_delta_risk(self, lr, theta, risk, risk_dontknow):
        """
        calculated expected risk reduction, i.e. int_{A2} R1(lr)-R2(lr)dlr
        where R1 is the risk of current optimal decision, R2 of alternative
        decision, A2 - region of optimality of alternative decision.
        Returns value normalized by current risk of optimal decision.

        lr - list of numpy arrays of likelihood ratios
        theta - current decision threshold
        risk - risk function of the decision (+ or -)
        risk_dontknow - risk_function of the "don't know" decision
        """
        mral = bootstrap_lr(
            lr, self.params['n_bootstrap'], trunc=self.params['truncation'])
        kde_mral = gaussian_kde(mral)

        grid = np.arange(0.1, mral.max() + 1, 0.01)
        dens_mral = kde_mral(grid)
        krit_idx = np.argmin(np.abs(grid - theta))

        mean_risk_decision = np.trapz(risk(grid) * dens_mral, x=grid)
        mean_risk_dontknow = np.trapz(risk_dontknow(grid) * dens_mral, x=grid)

        if mean_risk_decision < mean_risk_dontknow:
            delta_risk = self.risk_positive(
                grid[:krit_idx]) - self.risk_dontknow(grid[:krit_idx])
            delta_mean = np.trapz(
                dens_mral[:krit_idx] * delta_risk, x=grid[:krit_idx])
            rel_delta = delta_mean / mean_risk_decision
            return rel_delta, False
        else:
            delta_risk = self.risk_dontknow(
                grid[krit_idx:]) - self.risk_positive(grid[krit_idx:])
            delta_mean = np.trapz(
                dens_mral[krit_idx:] * delta_risk, x=grid[krit_idx:])
            rel_delta = delta_mean / mean_risk_dontknow
            return rel_delta, True

    def get_delta_convergence(self, lr, theta, risk, risk_dontknow):
        '''
   .    Examines whether delta risk has converged, i.e. whether additional
        measurements of the same sample still makes sense. If multiple 
        samples were given only last one is analyzed, i.e.  
        it is assumed that convergence of previous sample
        was already ensured before adding new sample
        lr - list of numpy arrays of likelihood ratios
        theta - current decision threshold
        risk - risk function of the decision (+ or -)
        risk_dontknow - risk_function of the "don't know" decision
        '''
        lr = lr[-1]
        run_deltas = []
        for _ in range(self.params['convergence_n_repeat']):
            idx = np.arange(lr.size)
            np.random.shuffle(idx)
            deltas = []
            for i in np.arange(lr.size - self.params["convergence_n_last"], lr.size, 1):
                drisk = self.get_rel_delta_risk(
                    (lr[idx[:i]],), theta, risk, risk_dontknow)
                deltas.append(drisk)
            run_deltas.append(deltas)
        res = np.array(run_deltas).std(0).mean()
        # print(f"Mean std: {res}")
        return res < self.params['convergence_threshold']

    def decide_samples(self, samples):
        '''
        Makes optimal decision for the given samples of spectra
        args:
        samples - list of numpy arrays containing spectra. Each array must contain
        spectra from single sample. Pass [sample] if want to predict spectr from one sample.
        '''
        if not isinstance(samples, list):
            raise ValueError(
                "sample must be a list of the arrays, not a single array!")
        emb = map(lambda x: self.embed(
            x, mult=self.params['n_samples_prediction']), samples)
        emb = tuple(map(self.remove_bad_spectra, emb))
        lr_positive, lr_negative = self.likelihood_ratio(emb)
        self.last_emb = emb
        print(self.decide(lr_positive, lr_negative))

    def visualize_sample(self):
        '''
        Plots distribution of the sample together with reference distibutions
        '''
        sns.distplot(self.emb_positive, label="POSITIVE",
                     hist=False, kde_kws={"shade": True})
        sns.distplot(self.emb_negative, label="NEGATIVE",
                     hist=False, kde_kws={"shade": True})
        for idx, sample in enumerate(self.last_emb):
            sns.distplot(sample.flatten(), label=f"SAMPLE{idx}",
                         hist=False, kde_kws={"shade": True})

    def visualize_intervals(self):
        grid = np.arange(self.interval_a, self.interval_b, 0.01)
        plt.plot(grid, self.spline_positive(grid), label="+")
        plt.plot(grid, self.spline_negative(grid), label="−")

        theta_positive_idx = np.argmin(np.abs(self.theta_positive -
                                              (self.spline_positive(grid) / self.spline_negative(grid))))
        theta_negative_idx = np.argmin(np.abs(self.theta_negative -
                                              (self.spline_negative(grid) / self.spline_positive(grid))))

        plt.axvline(x=grid[theta_positive_idx], color="k", linewidth=0.5)
        plt.axvline(x=grid[theta_negative_idx], color="k", linewidth=0.5)

        plt.fill_between(grid[:theta_negative_idx], self.spline_positive(
            grid[:theta_negative_idx]), color="r")
        plt.fill_between(grid[theta_positive_idx:], self.spline_negative(
            grid[theta_positive_idx:]), color="b")
        print(f"th+{self.theta_positive:.3f}; th-{self.theta_negative:.3f}")
        plt.legend()
        # plt.fill_between(grid[theta_negative_idx:theta_positive_idx],
        #                        self.spline_positive(grid[theta_negative_idx:theta_positive_idx]), color="g", alpha=0.5)

        # plt.fill_between(grid[theta_negative_idx:theta_positive_idx],
        #                        self.spline_negative(grid[theta_negative_idx:theta_positive_idx]), color="orange", alpha=0.5)

    def get_metrics(self):
        '''
        Calculates classification metrics like probability of false negative,
        false positive, false omission rate, i.e. P(FN|decicion=N),
        false detection rate i.e. P(FP|decision=P) and rejection rate,
        i.e. percent of samples rejected
        '''
        def lr_positive(x):
            return self.spline_positive(x) / self.spline_negative(x)

        def lr_negative(x):
            return self.spline_negative(x) / self.spline_positive(x)
        
        iqr = (self.iqr_positive + self.iqr_negative)/2
        
        try:
            theta_positive = brentq(lambda x: lr_positive(
                x) - self.theta_positive, self.interval_a - 0.5 * iqr, self.interval_b + 0.5 * iqr)
        except:
            theta_positive = self.interval_a - 0.5 * iqr
        try:
            theta_negative = brentq(lambda x: lr_negative(
                x) - self.theta_negative, self.interval_a - 0.5 * iqr, self.interval_b + 0.5 * iqr)
        except:
            theta_negative = self.interval_b + 0.5 * iqr

        fn_prob = quad(self.spline_positive,
                       self.interval_a, theta_negative)[0]
        tn_prob = quad(self.spline_negative,
                       self.interval_a, theta_negative)[0]

        fp_prob = quad(self.spline_negative,
                       theta_positive, self.interval_b)[0]
        tp_prob = quad(self.spline_positive,
                       theta_positive, self.interval_b)[0]

        rej_prob_positive = quad(self.spline_positive,
                                 theta_negative, theta_positive)[0]
        rej_prob_negative = quad(self.spline_negative,
                                 theta_negative, theta_positive)[0]

        P_FN = self.prior_positive * fn_prob / (fn_prob + tn_prob)
        P_FP = self.prior_negative * fp_prob / (fp_prob + tp_prob)
        P_rej = rej_prob_positive * self.prior_positive + \
            rej_prob_negative * self.prior_negative
        print(
            f"P(FN): {P_FN:.3f}; P(FP): {P_FP:.3f}; "
            f"P(rej): {P_rej:.3f}")
        return dict(P_FN=P_FN, P_FP=P_FP, P_rej=P_FP)
#%%


def custom_predict_step(self, data):
    data = data_adapter.expand_1d(data)
    x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
    return self(x, training=False)
