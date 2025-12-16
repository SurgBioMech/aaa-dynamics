import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
from scipy.special import logsumexp
import matplotlib as mpl
from aaa_org_helpers import *
from aaa_zsindy import *
from matplotlib import rcParams
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})
sns.set_style("white")

def make_kde_dict_cumulative_times(
    df,
    timepoints,
    value_cols=('SurfaceArea_Norm_mean', 'IntGaussian_Fluct_Norm_mean'),
    time_col='Years_mean',
):
    """
    For each time t in timepoints, build a KDE over value_cols using
    ALL points with time_col <= t (across all patients).
    """
    kde_dict = {}
    for t in timepoints:
        pts = df[df[time_col] <= t][list(value_cols)].to_numpy()
        pts = pts[~np.isnan(pts).any(axis=1)]
        kde_dict[t] = gaussian_kde(pts.T)
    return kde_dict

def static_bayes_class(
    regr_kde,
    stable_kde,
    regr_df,
    stable_df,
    t_min,
    t_max,
    prior=0.5,
    time_col='Years_mean',
    id_col='Patient_ID'
):
    """
    Static Bayesian classifier at time t_max using **history**.

    For each patient:
      - Collect all scans with t_min < time_col <= t_max.
      - For each scan k with state x_k, treat x_k as an independent draw
        from the KDE at this evaluation time (t_max).
      - Form the total likelihood:
            z_c = Π_k p(x_k | class=c, T=t_max)
        (implemented in log-space).
      - Combine with prior p(c) to get p(c | {x_k}, {t_k}).

    NOTE: This uses p(x_k | c, T=t_max) for all k, which is a practical
    approximation to Eq. (staticlikelihood). Exact Eq. would use p(x_k | c, t_k).
    """

    # Helper: select all scans in the window (t_min, t_max]
    def select_window(df):
        return df[(df[time_col] > t_min) & (df[time_col] <= t_max)].copy()

    regr_sel   = select_window(regr_df)
    regr_sel['ground_truth'] = 'Regr'

    stable_sel = select_window(stable_df)
    stable_sel['ground_truth'] = 'Stable'

    all_rows = pd.concat([regr_sel, stable_sel], ignore_index=True)

    # No data in this window at all
    if all_rows.empty:
        return pd.DataFrame(columns=[id_col, 'p_c2', 'ground_truth'])

    # If KDEs are missing, fall back to the prior for everyone
    if (regr_kde is None) or (stable_kde is None):
        # One row per patient, with prior probability
        base = (all_rows[[id_col, 'ground_truth']]
                .drop_duplicates(subset=[id_col])
                .reset_index(drop=True))
        base['p_c2'] = prior
        return base[[id_col, 'p_c2', 'ground_truth']]

    log_prior_regr   = np.log(1.0 - prior)
    log_prior_stable = np.log(prior)

    out_rows = []

    for pid, grp in all_rows.groupby(id_col):
        # all scans for this patient in (t_min, t_max]
        X = grp[['SurfaceArea_Norm_mean', 'IntGaussian_Fluct_Norm_mean']].to_numpy()

        # Defensive: if somehow no scans, skip
        if X.shape[0] == 0:
            out_rows.append({
                id_col: pid,
                'p_c2': prior,
                'ground_truth': grp['ground_truth'].iloc[0]
            })
            continue

        # Compute scan-wise likelihoods at this evaluation time
        # (using cumulative KDE at t_max for each scan)
        p_regr   = np.clip([regr_kde.pdf(x)   for x in X], 1e-12, None)
        p_stable = np.clip([stable_kde.pdf(x) for x in X], 1e-12, None)

        # Total log-likelihoods over history (product in log-space)
        log_like_regr   = np.sum(np.log(p_regr))
        log_like_stable = np.sum(np.log(p_stable))

        # Add log-priors
        lv_regr   = log_like_regr   + log_prior_regr
        lv_stable = log_like_stable + log_prior_stable

        # Normalize to get posterior
        lv = np.array([lv_regr, lv_stable])
        lv -= logsumexp(lv)
        post = np.exp(lv)

        out_rows.append({
            id_col: pid,
            'p_c2': post[1],  # probability of Stable
            'ground_truth': grp['ground_truth'].iloc[0]
        })

    return pd.DataFrame(out_rows)

def jitter_plot(
    regr_df,
    stable_df,
    timepoints,
    dfs,
    acc_bounds=(0.1, 0.9),
    axis_title=None,
    color_map=None,
    alpha_points=0.50, 
    dpi=100
):
    """
    Standalone jitter plot that matches MAIN_FIG2 posterior panels
    (colors, fonts, bands, alpha, etc.) but only reports % Classified.
    """
    # Default colors to match plot_posterior_scatters / MAIN_FIG2
    if color_map is None:
        color_map = {'Regr': '#1f77b4', 'Stable': '#d62728'}

    # Helper: force Times New Roman tick labels (like _force_tnr_ticks)
    def _force_tnr_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)
        for lab in ax.get_yticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)

    # Combine to get patient/scan counts by timepoint
    result_static = pd.concat([regr_df, stable_df], ignore_index=True)
    results_static = [pd.DataFrame({'Patient_ID': []})]
    for tp in timepoints:
        results_static.append(result_static[result_static['Years_mean'] <= tp])

    # Match MAIN_FIG2 y-labels
    labels = [
        'EVAR',
        f'{timepoints[0]} Yr.',
        f'{timepoints[1]} Yr.',
        f'{timepoints[2]} Yr.'
    ]

    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True, facecolor='white', dpi=dpi)
    low, high = acc_bounds

    for ax, df_t, lab, r in zip(axes, dfs, labels, results_static):
        df_t = df_t.copy()

        if df_t.empty:
            # Empty panel: just style the axes and continue
            ax.set_yticks([])
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(
                lab,
                rotation=0,
                labelpad=35,
                va='center',
                fontsize=12,
                fontname="Times New Roman"
            )
            _force_tnr_ticks(ax)
            continue

        df_t = df_t.reset_index(drop=True)

        # Jitter in [0,1]
        y_jit = np.random.rand(len(df_t))

        # Same front/back logic as plot_posterior_scatters
        blue_left   = (df_t['ground_truth'] == 'Regr')   & (df_t['p_c2'] < 0.5)
        blue_right  = (df_t['ground_truth'] == 'Regr')   & (df_t['p_c2'] >= 0.5)
        red_left    = (df_t['ground_truth'] == 'Stable') & (df_t['p_c2'] < 0.5)
        red_right   = (df_t['ground_truth'] == 'Stable') & (df_t['p_c2'] >= 0.5)
        grey = (df_t['p_c2'] == 0.5)

        # Draw “back” vs “front” layers
        ax.scatter(df_t.loc[red_left,  'p_c2'], y_jit[red_left],
                   c=color_map['Stable'], alpha=alpha_points, s=30, zorder=5)
        ax.scatter(df_t.loc[blue_left, 'p_c2'], y_jit[blue_left],
                   c=color_map['Regr'],   alpha=alpha_points, s=30, zorder=10)
        ax.scatter(df_t.loc[blue_right, 'p_c2'], y_jit[blue_right],
                   c=color_map['Regr'],   alpha=alpha_points, s=30, zorder=5)
        ax.scatter(df_t.loc[red_right,  'p_c2'], y_jit[red_right],
                   c=color_map['Stable'], alpha=alpha_points, s=30, zorder=10)
        ax.scatter(df_t.loc[grey, 'p_c2'], y_jit[grey],
                   c='grey', alpha=1, s=30, zorder=15)

        ax.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_facecolor('white')

        ax.set_ylabel(
            lab,
            rotation=0,
            labelpad=35,
            va='center',
            fontsize=12,
            fontname="Times New Roman"
        )

        # Vertical decision bounds (as in plot_posterior_scatters)
        ax.axvline(low,  color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        ax.axvline(high, color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)

        # Background classification bands (as in restyle_classifier_axes)
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle(
            (0.0, y0), low, y1 - y0,
            facecolor='#1f77b4', alpha=0.3, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (low, y0), high - low, y1 - y0,
            facecolor="#7e7e7e", alpha=0.45, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (high, y0), 1.0 - high, y1 - y0,
            facecolor='#d62728', alpha=0.3, lw=0, zorder=0
        ))

        # Ensure collections don't get too opaque
        for coll in ax.collections:
            try:
                a = coll.get_alpha()
                if a is None:
                    coll.set_alpha(alpha_points)
                else:
                    coll.set_alpha(min(a, alpha_points))
            except Exception:
                pass

        # -------- Metric: % Classified only --------
        df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
        n_total = len(df_valid)

        if n_total > 0:
            mask_conf_any = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)
            n_classified = mask_conf_any.sum()
            class_pct = (n_classified / n_total) * 100.0
        else:
            class_pct = np.nan

        class_str = (
            f"Classified: {class_pct:.1f}%"
            if not np.isnan(class_pct) else
            "Classified: n/a"
        )

        # Patient / scan counts based on cumulative result_static
        if len(r) > 0:
            patient_str = f"Patients: {len(r['Patient_ID'].unique())}"
            if len(r) < 200:
                scans_str   = f"{len(r)} CT scans\n75 real patients"
            else:
                scans_str   = f"{len(r)} CT scans + FEA points\n75 real patients"
        else:
            patient_str = "Patients: 0"
            if len(r) < 200:
                scans_str   = "75 patients pre-EVAR"

        # Centered boxed text like restyle_classifier_axes creates
        ax.text(
            0.5, 0.5,
            f"{class_str}\n{scans_str}",
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=12,
            fontname="Times New Roman",
            bbox=dict(facecolor='white', edgecolor='0.5',
                      boxstyle='round,pad=0.25'),
            zorder=100
        )

        # Ticks: only bottom panel shows x-ticks/labels
        if ax is not axes[-1]:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            if axis_title is not None:
                ax.set_xlabel(axis_title, fontsize=14, fontname="Times New Roman")
            else:
                ax.set_xlabel("p(Stable)", fontsize=14, fontname="Times New Roman")

        _force_tnr_ticks(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def stress_test_transformer(
    df: pd.DataFrame,
    *,
    id_col: str = 'Patient_ID',
    time_col: str = 'Years_mean',
    feature_cols=('SurfaceArea_Norm_mean', 'IntGaussian_Fluct_Norm_mean'),
    fixed_interval: float | None = 1.0,
    random_interval_range: tuple[float, float] | None = None,
    add_noise: bool = False,
    noise_pct: float = 0.0,
    keep_other_cols: bool = False,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Resample each patient's trajectory to a new time grid that ALWAYS starts at t=0,
    and optionally add percent-based noise to spatial features.

    Modes (choose exactly one):
      - Fixed interval:
            fixed_interval = Δt, random_interval_range = None
        => new times: 0, Δt, 2Δt, ... ≤ (t_max - t_min)
      - Random intervals:
            fixed_interval = None, random_interval_range = (low, high)
        => new times: 0, 0+U(low,high), 0+U(low,high)+U(low,high), ... ≤ (t_max - t_min)

    For each patient:
      1) Sort by `time_col`.
      2) Shift original times so earliest time becomes 0:
            t_shift = t_orig - min(t_orig)
      3) Build a new time grid (fixed or random) on [0, max(t_shift)].
      4) Linearly interpolate `feature_cols` onto that new grid.
      5) Optionally add multiplicative Gaussian noise to `feature_cols`.

    Returns:
      New DataFrame with resampled/perturbed trajectories.
    """
    if (fixed_interval is None) == (random_interval_range is None):
        raise ValueError(
            "Specify exactly one of `fixed_interval` or `random_interval_range`."
        )

    rng = np.random.default_rng(random_state)
    out_frames = []

    for pid, g in df.groupby(id_col):
        g = g.sort_values(time_col).copy()
        if g.empty:
            continue

        # Original times
        t_orig = g[time_col].to_numpy()
        t0 = t_orig[0]                 # earliest time for this patient
        t_shift = t_orig - t0          # shift so earliest becomes 0
        t_max_shift = t_shift[-1]      # max shifted time

        if t_max_shift < 0:
            # Shouldn't happen if times sorted, but guard anyway
            continue

        # --- build new time grid on [0, t_max_shift] ---
        if fixed_interval is not None:
            dt = float(fixed_interval)
            if dt <= 0:
                raise ValueError("fixed_interval must be positive.")
            # inclusive of t_max_shift within floating-point tolerance
            t_grid = np.arange(0.0, t_max_shift + 1e-9, dt)
        else:
            low, high = map(float, random_interval_range)
            if low <= 0 or high <= 0 or high < low:
                raise ValueError(
                    "random_interval_range must be (low, high) with 0 < low <= high."
                )
            t_list = [0.0]
            t = 0.0
            while True:
                t += rng.uniform(low, high)
                if t > t_max_shift:
                    break
                t_list.append(t)
            t_grid = np.array(t_list, dtype=float)

        # If for some reason we ended up with no times (e.g. single-point traj and random mode),
        # fall back to a single sample at t=0
        if t_grid.size == 0:
            t_grid = np.array([0.0], dtype=float)

        # --- interpolate features onto new grid ---
        data = {
            id_col: np.repeat(pid, len(t_grid)),
            time_col: t_grid  # NEW time axis, anchored at 0
        }

        for col in feature_cols:
            if col not in g.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            y_orig = g[col].to_numpy()
            # Interpolate on shifted domain
            data[col] = np.interp(t_grid, t_shift, y_orig)

        # broadcast other columns if requested (e.g., Label_mean, cohort, etc.)
        if keep_other_cols:
            const_cols = [
                c for c in g.columns
                if c not in (time_col,) and c not in feature_cols and c != id_col
            ]
            const_vals = g.iloc[0][const_cols]
            for c in const_cols:
                data[c] = np.repeat(const_vals[c], len(t_grid))

        out_frames.append(pd.DataFrame(data))

    if not out_frames:
        return df.iloc[0:0].copy()

    df_out = pd.concat(out_frames, ignore_index=True)

    # --- add spatial noise if requested ---
    if add_noise and noise_pct > 0:
        for col in feature_cols:
            vals = df_out[col].to_numpy()
            sigma = noise_pct * np.abs(vals)
            noise = rng.normal(loc=0.0, scale=sigma, size=len(vals))
            df_out[col] = vals + noise

    return df_out

def dynamic_bayes_class(
    results,
    regr_df,
    stable_df,
    t_min,
    t_max,
    *,
    prior=0.5,
    id_col='Patient_ID',
    time_col='Years_mean',
    verbose_fd=False,
    static_kde_dicts=None,   # tuple: (kde_regr_dict, kde_stable_dict) or None
    static_round_to=None     # e.g. 1.0 to round times to year grid
):
    """
    Combined Dynamic + Static Bayesian classifier over two classes (Regr, Stable).

    For each patient:
      - Consider all scans with t_min < time_col <= t_max (with small tolerance).
      - Build a trajectory in (A, K) space:
            A = SurfaceArea_Norm_mean
            K = IntGaussian_Fluct_Norm_mean
      - Dynamic term:
          * compute empirical derivatives via compute_finite_differences
          * for each class model c:
              - simulate SINDy model on the patient's time grid
              - compute model-predicted derivatives
              - compute Gaussian log-likelihood log z_c^dyn from SSE
      - Static term (optional, if static_kde_dicts is provided):
          * for each scan (x_k, t_k) in this window:
              - evaluate KDE p(x_k | c, t_k) (with rounding/fallback)
              - accumulate log-likelihood log z_c^stat = sum_k log p(...)
      - Total log-likelihood for class c:
            log z_c = log z_c^dyn + log z_c^stat
        (if no static_kde_dicts are given, log z_c^stat = 0)

      - Combine with priors (prior = p(Stable), 1-prior = p(Regr)):
            p(c | data) ∝ z_c * p(c)
        and normalize.

    Returns one row per patient:
        [Patient_ID, p_c2, ground_truth]
    where p_c2 = p(Stable | data).
    """

    # ----- ground truth tagging -----
    df_r = regr_df.copy()
    df_r['ground_truth'] = 'Regr'

    df_s = stable_df.copy()
    df_s['ground_truth'] = 'Stable'

    all_rows = pd.concat([df_r, df_s], ignore_index=True)

    # time window helper (keep slight tolerance like before)
    def select_window(g):
        return g[(g[time_col] > t_min - 1/12) &
                 (g[time_col] <= t_max + 1/12)].copy()

    # unpack dynamic models
    model_regr_dict, model_stable_dict = results
    two_models = [model_regr_dict, model_stable_dict]
    rho_list   = [model_regr_dict['rho'], model_stable_dict['rho']]

    # priors: prior = p(Stable), (1-prior) = p(Regr)
    log_prior_regr   = np.log(1.0 - prior)
    log_prior_stable = np.log(prior)

    # optional static KDEs: (kde_regr_dict, kde_stable_dict)
    use_static = static_kde_dicts is not None
    if use_static:
        kde_regr_dict, kde_stable_dict = static_kde_dicts
        kde_dicts = [kde_regr_dict, kde_stable_dict]
    else:
        kde_dicts = [None, None]

    def _static_loglik_for_patient(X, t_list, kde_by_time, round_to=None):
        """
        X: (n_scans, 2) array of [A, K]
        t_list: (n_scans,) array of times
        kde_by_time: dict {time -> gaussian_kde or None}
        Returns scalar log-likelihood sum_k log p(x_k | class, t_k).
        """
        if kde_by_time is None:
            return 0.0  # no static term

        logliks = []
        # prefilter available times with valid KDEs
        t_avail = np.array([tt for tt, v in kde_by_time.items() if v is not None])

        for x_k, t_k in zip(X, t_list):
            if t_avail.size == 0:
                # no KDEs at all → neutral / very weak contribution
                logliks.append(0.0)
                continue

            if round_to is not None:
                t_key = np.round(t_k / round_to) * round_to
                kde = kde_by_time.get(t_key, None)
                if kde is None:
                    # fallback: nearest available time
                    idx = np.argmin(np.abs(t_avail - t_k))
                    kde = kde_by_time[t_avail[idx]]
            else:
                # directly pick nearest available
                idx = np.argmin(np.abs(t_avail - t_k))
                kde = kde_by_time[t_avail[idx]]

            if kde is None:
                logliks.append(0.0)
            else:
                p = np.clip(kde.pdf(x_k), 1e-12, None)
                logliks.append(np.log(p))

        return float(np.sum(logliks)) if logliks else 0.0

    out = []

    for pid, grp_full in all_rows.groupby(id_col):
        grp = select_window(grp_full).sort_values(time_col)

        if grp.empty:
            # no data in this window → posterior = prior
            gt = grp_full['ground_truth'].iloc[0]
            out.append({
                id_col: pid,
                'p_c2': prior,
                'ground_truth': gt
            })
            continue

        # -------- spatial history --------
        comp = grp[[
            'SurfaceArea_Norm_mean',
            'IntGaussian_Fluct_Norm_mean',
            id_col,
            time_col,
            'ground_truth'
        ]].copy()

        comp['A_pred'] = comp['SurfaceArea_Norm_mean']
        comp['K_pred'] = comp['IntGaussian_Fluct_Norm_mean']

        # spatial positions and times
        X_spatial = comp[['SurfaceArea_Norm_mean',
                          'IntGaussian_Fluct_Norm_mean']].to_numpy()
        t_vals = comp[time_col].to_numpy()

        # -------- dynamic term: empirical derivatives --------
        xdot_true_df = compute_finite_differences(
            comp,
            verbose=verbose_fd
        )

        if xdot_true_df.empty:
            # derivatives wiped out by cleaning → use ONLY static term (if available)
            log_z = np.zeros(2)
            for j in range(2):
                log_dyn = 0.0
                log_stat = _static_loglik_for_patient(
                    X_spatial, t_vals, kde_dicts[j], round_to=static_round_to
                ) if use_static else 0.0
                log_z[j] = log_dyn + log_stat

            lv = np.array([
                log_z[0] + log_prior_regr,
                log_z[1] + log_prior_stable
            ])
            lv -= logsumexp(lv)
            post = np.exp(lv)

            gt = grp['ground_truth'].iloc[0]
            out.append({
                id_col: pid,
                'p_c2': post[1],
                'ground_truth': gt
            })
            continue

        idx_eff = xdot_true_df.index
        xdot_true = xdot_true_df[[
            'dSurfaceArea_Norm_mean/dYears_mean',
            'dIntGaussian_Fluct_Norm_mean/dYears_mean'
        ]].to_numpy()

        n_eff = xdot_true.shape[0]
        d_dim = xdot_true.shape[1]  # should be 2

        # -------- combine dynamic + static for each class --------
        log_z = np.zeros(2)

        for j, mdl in enumerate(two_models):
            model_j = mdl['model']
            x0_j    = mdl['x0']
            rho_j   = rho_list[j]

            # simulate class model on this patient's time grid
            x_pred = model_j.simulate(x0_j, t_vals)
            comp['A_pred'], comp['K_pred'] = x_pred[:, 0], x_pred[:, 1]

            # predicted derivatives on same FD scheme
            xdot_pred_df = compute_finite_differences(
                comp,
                verbose=False
            )

            # align with xdot_true via idx_eff
            xdot_pred = xdot_pred_df.loc[idx_eff, [
                'dA_pred/dYears_mean',
                'dK_pred/dYears_mean'
            ]].to_numpy()

            # dynamic SSE
            SSE = np.sum((xdot_true - xdot_pred) ** 2)

            # Gaussian dynamic log-likelihood
            log_norm  = - (d_dim * n_eff / 2.0) * np.log(2.0 * np.pi * (rho_j ** 2))
            log_resid = - SSE / (2.0 * (rho_j ** 2))
            log_dyn   = log_norm + log_resid

            # static spatial log-likelihood (if KDEs provided)
            log_stat = _static_loglik_for_patient(
                X_spatial, t_vals, kde_dicts[j], round_to=static_round_to
            ) if use_static else 0.0

            log_z[j] = log_dyn + log_stat

        # priors + normalization
        lv = np.array([
            log_z[0] + log_prior_regr,
            log_z[1] + log_prior_stable
        ])
        lv -= logsumexp(lv)
        post = np.exp(lv)

        gt = grp['ground_truth'].iloc[0]
        out.append({
            id_col: pid,
            'p_c2': post[1],
            'ground_truth': gt
        })

    return pd.DataFrame(out)

def plot_posterior_scatters(
    fig,
    outer,
    row,
    dfs,
    *,
    col_idx=4,
    title=None,
    xlabel=None,
    acc_bounds=(0.1, 0.9),
    color_map=None,
):
    """
    Add vertically stacked posterior scatter plots in one column of the grid.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    outer : matplotlib.gridspec.GridSpec
        The outer GridSpec of the figure.
    row : int
        Row index in `outer` where the column should be placed.
    dfs : list of DataFrame
        dfs[i] must have columns ['ground_truth', 'p_c2'] for each time panel.
        Typically length 4: [t0 prior, t1, t2, t3].
    col_idx : int, default 4
        Column index in outer GridSpec.
    title : str, optional
        Title for the top panel.
    xlabel : str, optional
        X-label for the bottom panel.
    acc_bounds : (float, float), default (0.1, 0.9)
        Thresholds for calling a point “correct”:
        - Stable: p_c2 > acc_bounds[1]
        - Regr:   p_c2 < acc_bounds[0]
    color_map : dict, optional
        Mapping {'Regr': color, 'Stable': color}. If None, defaults to blue/red.
    """
    if color_map is None:
        color_map = {'Regr': '#1f77b4', 'Stable': '#d62728'}

    n_panels = len(dfs)
    sub = outer[row, col_idx].subgridspec(n_panels, 1, hspace=0.2)

    for i in range(n_panels):
        ax = fig.add_subplot(sub[i])
        df_t = dfs[i].copy()
        if df_t.empty:
            ax.set_yticks([])
            ax.set_xlim(-0.05, 1.05)
            if i == n_panels - 1 and xlabel:
                ax.set_xlabel(xlabel, fontsize=14)
            if i == 0 and title:
                ax.set_title(title, fontsize=16, pad=10)
            continue

        df_t = df_t.reset_index(drop=True)
        y_jit = np.random.rand(len(df_t))

        blue_left   = (df_t['ground_truth'] == 'Regr')   & (df_t['p_c2'] < 0.5)
        blue_right  = (df_t['ground_truth'] == 'Regr')   & (df_t['p_c2'] >= 0.5)
        red_left    = (df_t['ground_truth'] == 'Stable') & (df_t['p_c2'] < 0.5)
        red_right   = (df_t['ground_truth'] == 'Stable') & (df_t['p_c2'] >= 0.5)

        # draw in back/front layers so “correct side” points come to the front
        ax.scatter(df_t.loc[red_left,  'p_c2'], y_jit[red_left],
                   c=color_map['Stable'], alpha=0.1, s=30, zorder=5)
        ax.scatter(df_t.loc[blue_left, 'p_c2'], y_jit[blue_left],
                   c=color_map['Regr'],   alpha=0.1, s=30, zorder=10)
        ax.scatter(df_t.loc[blue_right, 'p_c2'], y_jit[blue_right],
                   c=color_map['Regr'],   alpha=0.1, s=30, zorder=5)
        ax.scatter(df_t.loc[red_right,  'p_c2'], y_jit[red_right],
                   c=color_map['Stable'], alpha=0.1, s=30, zorder=10)

        ax.set_yticks([])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylabel("")

        # vertical decision bounds
        ax.axvline(acc_bounds[0], color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        ax.axvline(acc_bounds[1], color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)

        if i < n_panels - 1:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=14)

        if i == 0 and title:
            ax.set_title(title, fontsize=16, pad=10)

        if i > 0:
            # Use same definition as jitter plot:
            # Classified = points with p_c2 outside [acc_bounds[0], acc_bounds[1]]
            df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
            n_total = len(df_valid)

            if n_total > 0:
                classified_mask = (
                    (df_valid['p_c2'] < acc_bounds[0]) |
                    (df_valid['p_c2'] > acc_bounds[1])
                )
                n_classified = classified_mask.sum()
                class_pct = (n_classified / n_total) * 100.0
            else:
                class_pct = np.nan

            class_str = (
                f"Classified: {class_pct:.1f}%"
                if not np.isnan(class_pct) else
                "Classified: n/a"
            )

            # Initial placement; restyle_classifier_axes will re-center and box it
            ax.text(
                -0.01, 0.5,
                class_str,
                transform=ax.transAxes,
                ha='right', va='center',
                fontsize=13, color='k', rotation=0
            )
            
def SINDY_BAYES(
    datasets_regr,
    datasets_stable,
    dfs_static,
    dfs_dyn,
    *,
    poly_degree=1,
    max_terms=3,
    x0=None,
    x0_std=None,
    ens_trials=1000,
    colors_regr=('#1f77b4', '#1f77b4'),
    colors_stable=('#d62728', '#d62728'),
    density_cols_wspace=0.02,
    acc_bounds=(0.1, 0.9),
    dpi=100
):
    # helper: force Times New Roman on tick labels with hard-coded sizes
    def _force_tnr_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontname("Times New Roman"); lab.set_fontsize(12)
        for lab in ax.get_yticklabels():
            lab.set_fontname("Times New Roman"); lab.set_fontsize(12)

    def _normalize_datasets(ds, default_lmbda=-5.0, name_prefix=''):
        # Already list/tuple of dicts → assume caller did it right
        if isinstance(ds, (list, tuple)):
            return list(ds)
        # Single entry dict with 'df' → wrap
        if isinstance(ds, dict) and 'df' in ds:
            return [ds]
        # Bare DataFrame → wrap into the expected dict form
        if isinstance(ds, pd.DataFrame):
            return [{'df': ds, 'name': name_prefix, 'lmbda': default_lmbda}]
        raise TypeError(
            f"Unsupported dataset type {type(ds)}. "
            "Pass a DataFrame, a dict with 'df', or a list of such dicts."
        )

    datasets_regr = _normalize_datasets(datasets_regr, default_lmbda=-5.0, name_prefix='Regr')
    datasets_stable = _normalize_datasets(datasets_stable, default_lmbda=-5.0, name_prefix='Stable')


    assert len(datasets_regr) == len(datasets_stable), "Lists must match length"
    N = len(datasets_regr)

    def process_list(ds_list):
        res = []
        for entry in ds_list:
            df = entry['df'].sort_values('Years_mean')
            name = entry.get('name', None)

            x = df[['SurfaceArea_Norm_mean',
                    'IntGaussian_Fluct_Norm_mean']].to_numpy()
            t = df['Years_mean'].to_numpy()

            xdot_cols = [
                'dSurfaceArea_Norm_mean/dYears_mean',
                'dIntGaussian_Fluct_Norm_mean/dYears_mean'
            ]
            xdot = df[xdot_cols].to_numpy()

            x0u = x0 if x0 is not None else x[0]
            mu, _, rho = estimate_rho(x, xdot)
            A_fp, K_fp = get_fixed_points(mu)

            model = ZSindy(
                poly_degree=poly_degree,
                lmbda=entry['lmbda'],
                max_num_terms=max_terms,
                rho=rho,
                variable_names=['A', 'K']
            )
            model.fit(x, t, xdot)
            coef = model.coefficients()
            var  = model.coefficients_variance()

            sol = solve_ivp(
                lambda tt, yy: DiffEq(tt, yy, coef),
                (0, 10), x0u,
                t_eval=np.linspace(0, 10, 500)
            )

            res.append({
                'name': name, 'df': df,
                't': t, 'x': x, 'xdot': xdot,
                'sol': sol,
                'coef': coef, 'sd': np.sqrt(var),
                'A_fp': A_fp, 'K_fp': K_fp,
                'model': model, 'x0': x0u, 'rho': rho,
                'mu': mu, 'B': mu[1:, :].T
            })
        return res

    res_r = process_list(datasets_regr)
    res_s = process_list(datasets_stable)

    timepoints = [0, 1, 2, 5]
    ylabels = {0: 'EVAR', 1: '1 Yr.', 2: '2 Yr.', 5: '5 Yr.'}

    fig = plt.figure(figsize=(16, 5 * N), facecolor='white', dpi=dpi)
    outer = gridspec.GridSpec(
        N, 6, figure=fig,
        width_ratios=[1, 0.5, 0.5, 0.5, 0.75, 0.75],
        hspace=0.55,
        wspace=0.20
    )

    def simulate(res):
        sol  = res['sol']
        coef = res['coef']
        sd   = res['sd']
        x0u  = res['x0']
        model = res['model']

        if x0_std is None:
            x0s = np.tile(x0u, (ens_trials, 1))
        else:
            x0s = np.random.normal(x0u, x0_std, size=(ens_trials, len(x0u)))

        nd, nf = coef.shape
        cs = np.random.normal(
            coef.ravel(), sd.ravel(),
            size=(ens_trials, nd * nf)
        ).reshape(ens_trials, nd, nf)

        ens = np.stack([model.simulate(x0s[k], sol.t, cs[k])
                        for k in range(ens_trials)])
        return sol, ens

    def restyle_classifier_axes(ax, alpha_points=0.10):
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle((0.0, y0), 0.1, y1 - y0,
                               facecolor='#1f77b4', alpha=0.3, lw=0, zorder=0))
        ax.add_patch(Rectangle((0.1, y0), 0.8, y1 - y0,
                               facecolor="#7e7e7e",   alpha=0.45, lw=0, zorder=0))
        ax.add_patch(Rectangle((0.9, y0), 0.1, y1 - y0,
                               facecolor='#d62728', alpha=0.3, lw=0, zorder=0))
        ax.set_xlim(0.0, 1.0)
        ax.set_facecolor('white')

        for coll in ax.collections:
            try:
                coll.set_alpha(min(coll.get_alpha() or 1.0, alpha_points))
            except Exception:
                pass

        # Find any "Classified:" text and restyle it into a centered boxed label
        for txt in list(ax.texts):
            s = txt.get_text()
            if 'Classified' in s:
                txt.set_position((0.5, 0.5))
                txt.set_transform(ax.transAxes)
                txt.set_ha('center')
                txt.set_va('center')
                txt.set_fontstyle('normal')
                txt.set_color('k')
                txt.set_bbox(dict(facecolor='white',
                                  edgecolor='0.5',
                                  boxstyle='round,pad=0.25'))
                txt.set_zorder(100)
                txt.set_fontsize(12)
                txt.set_fontname("Times New Roman")

        ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontname="Times New Roman")
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontname="Times New Roman",
                      labelpad=0)
        _force_tnr_ticks(ax)

        if ax.get_title():
            ax.set_title(ax.get_title(), fontname="Times New Roman", fontsize=16)

    def make_first_row_points_black(new_axes_in_col):
        if not new_axes_in_col:
            return
        first_ax = sorted(
            new_axes_in_col,
            key=lambda a: a.get_position().y1,
            reverse=True
        )[0]
        for coll in first_ax.collections:
            try:
                if hasattr(coll, "get_offsets"):
                    coll.set_facecolor("#7c7c7c")
            except Exception:
                pass

    # ----- main rows -----
    for i in range(N):
        # Static classifier (col 4)
        ax_before = set(fig.axes)
        plot_posterior_scatters(
            fig, outer, i, dfs_static,
            col_idx=4,
            title="Static Classifier\n$(\\widetilde{A}, \\widetilde{\\delta K}) \\rightarrow$ Class",
            xlabel="p(Stable | Statics)",
            acc_bounds=(0.1, 0.9),
        )
        new_axes_static = [a for a in fig.axes if a not in ax_before]
        for ax in new_axes_static:
            restyle_classifier_axes(ax)
        make_first_row_points_black(new_axes_static)

        # Dynamic classifier (col 5)
        ax_before = set(fig.axes)
        plot_posterior_scatters(
            fig, outer, i, dfs_dyn,
            col_idx=5,
            title="Dynamic Classifier\n$(\\widetilde{A}, \\widetilde{\\delta K}, \\dot{\\widetilde{A}}, \\dot{\\widetilde{\\delta K}}) \\rightarrow$ Class",
            xlabel="p(Stable | Dynamics)",
            acc_bounds=acc_bounds,
        )
        new_axes_dyn = [a for a in fig.axes if a not in ax_before]
        for ax in new_axes_dyn:
            restyle_classifier_axes(ax)
        make_first_row_points_black(new_axes_dyn)

        # ensemble sims
        sol_r, ens_r = simulate(res_r[i])
        sol_s, ens_s = simulate(res_s[i])

        # Column 0: Ensemble Trajectories
        ens_gs = outer[i, 0].subgridspec(2, 1, hspace=0.22)

        axA = fig.add_subplot(ens_gs[0]); axA.set_facecolor('white')
        axA.axvline(0, color='k', ls=':')
        axA.plot(sol_r.t, ens_r[:, :, 0].T, color=colors_regr[0], alpha=0.01)
        axA.plot(sol_s.t, ens_s[:, :, 0].T, color=colors_stable[0], alpha=0.01)
        axA.plot(sol_r.t, sol_r.y[0], '--k', lw=2)
        axA.plot(sol_s.t, sol_s.y[0], '--k', lw=2)
        axA.set_title('Ensemble Trajectories',
                      fontsize=16, fontname="Times New Roman")
        axA.set_ylabel(r'$\widetilde{A}$',
                       fontsize=16, fontname="Times New Roman")
        axA.set_xlabel('', fontname="Times New Roman")
        axA.set_xticks([0, 5, 10]); axA.set_xticklabels(['EVAR', '5', '10'])
        axA.set_ylim(0, 10)
        axA.set_yticks([0, 5, 10]); axA.set_yticklabels(['0', '5', '10'])
        _force_tnr_ticks(axA)
        legend_lines = [
            Line2D([0], [0], color=colors_regr[0], lw=2, label='Regressing Sac'),
            Line2D([0], [0], color=colors_stable[0], lw=2, label='Stable Sac'),
            Line2D([0], [0], color='k', lw=2, linestyle='--',
                   label='Mean Trajectory')
        ]
        leg = axA.legend(handles=legend_lines, loc='upper left', frameon=False)
        for t in leg.get_texts():
            t.set_fontsize(10); t.set_fontname("Times New Roman")

        axK = fig.add_subplot(ens_gs[1]); axK.set_facecolor('white')
        axK.axvline(0, color='k', ls=':')
        axK.plot(sol_r.t, ens_r[:, :, 1].T, color=colors_regr[1], alpha=0.01)
        axK.plot(sol_s.t, ens_s[:, :, 1].T, color=colors_stable[1], alpha=0.01)
        axK.plot(sol_r.t, sol_r.y[1], '--k', lw=2)
        axK.plot(sol_s.t, sol_s.y[1], '--k', lw=2)
        axK.set_ylabel(r'$\widetilde{\delta K}$',
                       fontsize=16, fontname="Times New Roman", labelpad=0)
        axK.set_xlabel('Years',
                       fontsize=16, fontname="Times New Roman")
        axK.set_xticks([0, 5, 10]); axK.set_xticklabels(['EVAR', '5', '10'])
        axK.set_ylim(0, 10)
        axK.set_yticks([0, 5, 10]); axK.set_yticklabels(['0', '5', '10'])
        _force_tnr_ticks(axK)

        # Columns 1–3: 1D & 2D density plots
        density_top_axes = []
        for col_idx, dim in enumerate(['A', 'K', '2D'], start=1):
            sub = outer[i, col_idx].subgridspec(
                len(timepoints), 1,
                hspace=0.08, wspace=density_cols_wspace
            )
            for k, tp in enumerate(timepoints):
                ax = fig.add_subplot(sub[k]); ax.set_facecolor('white')
                idx = np.argmin(np.abs(sol_r.t - tp))
                if dim == 'A':
                    sns.kdeplot(ens_r[:, idx, 0], ax=ax, color=colors_regr[0])
                    sns.kdeplot(ens_s[:, idx, 0], ax=ax, color=colors_stable[0])
                    ax.set_xlim(0, 7.5)
                    ax.set_xticks([0, 7.5]); ax.set_xticklabels(['0', '7.5'])
                    ax.set_yticks([])
                elif dim == 'K':
                    sns.kdeplot(ens_r[:, idx, 1], ax=ax, color=colors_regr[1])
                    sns.kdeplot(ens_s[:, idx, 1], ax=ax, color=colors_stable[1])
                    ax.set_xlim(0, 10)
                    ax.set_xticks([0, 10]); ax.set_xticklabels(['0', '10'])
                    ax.set_yticks([])
                else:
                    sns.scatterplot(
                        x=ens_r[:, idx, 0], y=ens_r[:, idx, 1],
                        color=colors_regr[0], alpha=0.025,
                        ax=ax, edgecolor=None
                    )
                    sns.scatterplot(
                        x=ens_s[:, idx, 0], y=ens_s[:, idx, 1],
                        color=colors_stable[0], alpha=0.025,
                        ax=ax, edgecolor=None
                    )
                    sns.kdeplot(
                        x=ens_r[:, idx, 0], y=ens_r[:, idx, 1],
                        levels=[0.5, 0.8],
                        color=colors_regr[0], ax=ax
                    )
                    sns.kdeplot(
                        x=ens_s[:, idx, 0], y=ens_s[:, idx, 1],
                        levels=[0.5, 0.8],
                        color=colors_stable[0], ax=ax
                    )
                    ax.set_xlim(0, 7.5)
                    ax.set_xticks([0, 7.5]); ax.set_xticklabels(['0', '7.5'])
                    ax.set_ylim(0, 10); ax.set_yticks([])
                    ax.yaxis.labelpad = 0

                if col_idx == 1:
                    ax.set_ylabel(ylabels[tp],
                                  fontsize=12, fontname="Times New Roman")
                else:
                    ax.set_ylabel(
                        r'$\widetilde{\delta K}$' if col_idx == 3 else '',
                        fontsize=12, fontname="Times New Roman", labelpad=0
                    )

                if k == len(timepoints) - 1:
                    if col_idx == 1:
                        ax.set_xlabel(r'$\widetilde{A}$',
                                      fontsize=12, fontname="Times New Roman")
                    elif col_idx == 2:
                        ax.set_xlabel(r'$\widetilde{\delta K}$',
                                      fontsize=12, fontname="Times New Roman")
                    elif col_idx == 3:
                        ax.set_xlabel(r'$\widetilde{A}$',
                                      fontsize=12, fontname="Times New Roman")
                else:
                    ax.set_xticks([])

                _force_tnr_ticks(ax)

                if k == 0:
                    density_top_axes.append(ax)

        if len(density_top_axes) >= 3:
            lp = min(a.get_position().x0 for a in density_top_axes[0:3])
            rp = max(a.get_position().x1 for a in density_top_axes[0:3])
            yp = max(a.get_position().y1 for a in density_top_axes[0:3])
            fig.text(
                (lp + rp) / 2, yp + 0.018,
                "1D and 2D Density Plots",
                ha='center', va='bottom',
                fontsize=16, fontname="Times New Roman"
            )

    plt.tight_layout()
    plt.show()
    return res_r + res_s

def run_stress_test_analysis(
    AK_summary,
    regr_all,
    stable_all,
    noise_list      = (0.0, 0.1, 0.25),
    interval_list   = (1.0, None, None),
    int_range_list  = (None, (0.5, 1.5), (1.0, 3.0)),
    max_time        = 5.0,
    n_realizations  = 5,
    random_seed     = 0,
    acc_bounds      = (0.1, 0.9),
):
    """
    Run stress-test experiments over different noise levels and sampling schedules.

    Returns
    -------
    results : dict
        keys: (noise_pct, schedule_idx)
        values: DataFrame with columns:
            - time
            - accuracy      : mean accuracy over realizations (fraction)
            - accuracy_std  : std of accuracy over realizations
            - classified    : mean fraction classified over realizations
            - classified_std: std of fraction classified over realizations
    """

    # ---- sanity check ----
    if len(interval_list) != len(int_range_list):
        raise ValueError("interval_list and int_range_list must have same length.")

    rng = np.random.default_rng(random_seed)
    low, high = acc_bounds

    # common evaluation time grid (classification times)
    eval_times = np.arange(0.0, max_time + 1e-9, 1.0)  # 0,1,2,...,max_time

    # ---------- precompute ensembles ONCE ----------

    best_results_base, regr_ens_base, stable_ens_base = zsindy_pipeline(
        [{'df': regr_all, 'name': '', 'lmbda': -5},],
        [{'df': stable_all, 'name': '', 'lmbda': -5},],
        x0     = [AK_summary.iloc[0, 0], AK_summary.iloc[0, 2]],
        x0_std = [AK_summary.iloc[0, 1] / 3, AK_summary.iloc[0, 3] / 3],
        colors = colors,
        plot_bool=False
    )

    # give synthetic "patient IDs" per ensemble trial
    regr_ens_base   = regr_ens_base.copy()
    stable_ens_base = stable_ens_base.copy()

    regr_ens_base['Patient_ID']   = 'REGR'   + regr_ens_base['trial'].astype(str)
    stable_ens_base['Patient_ID'] = 'STABLE' + stable_ens_base['trial'].astype(str)

    regr_ens_base   = regr_ens_base.sort_values(['Patient_ID', 'Years_mean']).copy()
    stable_ens_base = stable_ens_base.sort_values(['Patient_ID', 'Years_mean']).copy()

    regr_ens_base['order']   = regr_ens_base.groupby('Patient_ID').cumcount() + 1
    stable_ens_base['order'] = stable_ens_base.groupby('Patient_ID').cumcount() + 1

    regr_ens_base['Scan_ID']   = regr_ens_base['Patient_ID']   + '_' + regr_ens_base['order'].astype(str)
    stable_ens_base['Scan_ID'] = stable_ens_base['Patient_ID'] + '_' + stable_ens_base['order'].astype(str)

    # ---------- evaluate across settings using the same base ensembles ----------

    results = {}

    total_iters = len(interval_list) * len(noise_list) * n_realizations * len(eval_times)

    with tqdm(total=total_iters, desc="Running experiments", leave=True) as pbar:

        for sched_idx, (interval, int_range) in enumerate(zip(interval_list, int_range_list)):

            # decide which scheduling mode
            if interval is not None and int_range is not None:
                raise ValueError("For each schedule, use either fixed interval or random range, not both.")
            fixed_interval = interval
            random_range   = int_range

            runs = []  # rows: (sched_idx, noise_pct, rep, time, accuracy, classified)

            for noise_pct in noise_list:

                for rep in range(n_realizations):

                    # per-(schedule,noise,rep) random seed so it's reproducible
                    rs = int(rng.integers(0, 1_000_000))

                    # --- transform regression / stable ensembles with spacing/noise ---
                    data_r = stress_test_transformer(
                        regr_ens_base,
                        id_col='Patient_ID',
                        time_col='Years_mean',
                        feature_cols=('SurfaceArea_Norm_mean', 'IntGaussian_Fluct_Norm_mean'),
                        fixed_interval=fixed_interval,
                        random_interval_range=random_range,
                        add_noise=(noise_pct > 0),
                        noise_pct=noise_pct,
                        keep_other_cols=True,
                        random_state=rs,
                    )
                    data_s = stress_test_transformer(
                        stable_ens_base,
                        id_col='Patient_ID',
                        time_col='Years_mean',
                        feature_cols=('SurfaceArea_Norm_mean', 'IntGaussian_Fluct_Norm_mean'),
                        fixed_interval=fixed_interval,
                        random_interval_range=random_range,
                        add_noise=(noise_pct > 0),
                        noise_pct=noise_pct,
                        keep_other_cols=True,
                        random_state=rs + 1,  # different noise than data_r
                    )

                    # ground-truth labels per synthetic "patient"
                    df0_r = data_r.drop_duplicates('Patient_ID')[['Patient_ID']].assign(ground_truth='Regr')
                    df0_s = data_s.drop_duplicates('Patient_ID')[['Patient_ID']].assign(ground_truth='Stable')
                    ground_truth_df = (
                        pd.concat([df0_r, df0_s], ignore_index=True)
                          .drop_duplicates('Patient_ID')
                    )

                    for t in eval_times:
                        if t == 0:
                            # prior: all p_c2 = 0.5 (completely uninformative)
                            df_t = ground_truth_df.copy()
                            df_t['p_c2'] = 0.5
                        else:
                            kde_regr   = make_kde_dict_cumulative_times(data_r, [t])
                            kde_stable = make_kde_dict_cumulative_times(data_s, [t])

                            df_t = dynamic_bayes_class(
                                results          = best_results_base,
                                regr_df          = data_r,
                                stable_df        = data_s,
                                t_min            = 0.0,
                                t_max            = t,
                                prior            = 0.5,
                                id_col           = 'Patient_ID',
                                time_col         = 'Years_mean',
                                verbose_fd       = False,
                                static_kde_dicts = (kde_regr, kde_stable)
                            )

                        # ----- metrics: accuracy & classified% (jitter-plot style) -----
                        if len(df_t) > 0:
                            df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
                            n_total = len(df_valid)

                            if n_total > 0:
                                classified_mask = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)

                                correct_mask = (
                                    (df_valid['ground_truth'] == 'Stable') & (df_valid['p_c2'] > high)
                                ) | (
                                    (df_valid['ground_truth'] == 'Regr') & (df_valid['p_c2'] < low)
                                )

                                n_classified = classified_mask.sum()
                                n_correct    = (classified_mask & correct_mask).sum()

                                if n_classified > 0:
                                    acc_frac = n_correct / n_classified
                                else:
                                    acc_frac = np.nan

                                class_frac = n_classified / n_total
                            else:
                                acc_frac   = np.nan
                                class_frac = np.nan
                        else:
                            acc_frac   = np.nan
                            class_frac = np.nan

                        runs.append((sched_idx, noise_pct, rep, t, acc_frac, class_frac))

                        pbar.update(1)

            runs_df = pd.DataFrame(
                runs,
                columns=['schedule_idx', 'noise', 'rep', 'time', 'accuracy', 'classified']
            )

            # summarize per noise level for this schedule
            for noise_pct in noise_list:
                sub = runs_df[runs_df['noise'] == noise_pct]
                if sub.empty:
                    continue

                summary = sub.groupby('time', as_index=False).agg(
                    accuracy      = ('accuracy', 'mean'),
                    accuracy_std  = ('accuracy', 'std'),
                    classified    = ('classified', 'mean'),
                    classified_std= ('classified', 'std'),
                )

                results[(noise_pct, sched_idx)] = summary[
                    ['time', 'accuracy', 'accuracy_std', 'classified', 'classified_std']
                ]

    return results


def plot_classifier_curves(results, figsize=(12,6), sd_mult=1.0, dpi=100):
    """
    Plot mean accuracy (left) and mean classified fraction (right)
    with the same colors/linestyles for noise and schedule.

    - Left: Accuracy of classified (%) with ±(sd_mult * acc_std) error bars
    - Right: Classified (%) with ±(sd_mult * classified_std) error bars
    """
    # ---------- flatten results dict ----------
    frames = []
    for (noise, interval), df in results.items():
        tmp = df.copy()
        tmp['noise'] = noise
        tmp['interval_str'] = str(interval)   # this should be schedule_idx (0,1,2)
        frames.append(tmp)
    results_df = pd.concat(frames, ignore_index=True)

    # ---------- colors for noise ----------
    unique_noises = sorted(results_df['noise'].unique())
    default_colors = ["#006144", "#bda000", "#911d00"]
    if len(unique_noises) <= len(default_colors):
        noise_palette = {n: default_colors[i] for i, n in enumerate(unique_noises)}
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', default_colors)
        noise_palette = {n: prop_cycle[i % len(prop_cycle)] for i, n in enumerate(unique_noises)}

    # ---------- distinct dash patterns ----------
    interval_styles = {'0': 'solid', '1': 'dashed', '2': 'dotted'}

    # two side-by-side subplots
    fig, (ax_cls, ax_acc) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    ax_cls.grid(True, linestyle='--', alpha=0.5)

    # ---------- plot curves on both axes ----------
    for (noise, interval_str), sub in results_df.groupby(['noise', 'interval_str']):
        color = noise_palette[noise]
        style = interval_styles.get(interval_str, 'solid')

        # Work on a copy to overwrite t=0
        sub_plot = sub.copy()

        # Force prior at t = 0:
        #   - accuracy: 0%
        #   - classified: 0%  (your request)
        mask_t0 = (sub_plot['time'] == 0)
        sub_plot.loc[mask_t0, 'accuracy']        = 0.0
        sub_plot.loc[mask_t0, 'accuracy_std']         = 0.0
        sub_plot.loc[mask_t0, 'classified']      = 0.0
        if 'classified_std' in sub_plot.columns:
            sub_plot.loc[mask_t0, 'classified_std'] = 0.0

        # Convert to %
        acc = sub_plot['accuracy'].astype(float) * 100.0
        acc_sd = sub_plot['accuracy_std'].fillna(0).astype(float) * 100.0
        yerr_acc = (sd_mult * acc_sd).to_numpy()

        classified = sub_plot['classified'].astype(float) * 100.0
        if 'classified_std' in sub_plot.columns:
            cls_sd = sub_plot['classified_std'].fillna(0).astype(float) * 100.0
        else:
            cls_sd = np.zeros_like(classified)
        yerr_cls = (sd_mult * cls_sd).to_numpy()

        # ----- left axis: accuracy of classified -----
        ax_acc.plot(
            sub_plot['time'], acc,
            linestyle=style, color=color, marker='o',
            lw=2, alpha=1, markersize=12
        )
        ax_acc.errorbar(
            sub_plot['time'], acc,
            yerr=yerr_acc,
            fmt='none', ecolor=color, elinewidth=3.0,
            capsize=4, alpha=0.8
        )

        # ----- right axis: classified percentage -----
        ax_cls.plot(
            sub_plot['time'], classified,
            linestyle=style, color=color, marker='o',
            lw=2, alpha=1, markersize=12
        )
        ax_cls.errorbar(
            sub_plot['time'], classified,
            yerr=yerr_cls,
            fmt='none', ecolor=color, elinewidth=3.0,
            capsize=4, alpha=0.8
        )

    # ---------- axes: left (accuracy) ----------
    ax_acc.set_xlabel("Years since EVAR", fontsize=20, labelpad=10)
    ax_acc.set_ylabel("Accuracy of Classified (%)", fontsize=20, labelpad=4)
    ax_acc.set_xlim(0, 5.5)
    ax_acc.set_ylim(-10, 110)
    ax_acc.set_xticks([0, 1, 2, 3, 4, 5])
    ax_acc.set_xticklabels(["EVAR", "1", "2", "3", "4", "5"])
    ax_acc.set_yticks([0, 25, 50, 75, 100])
    ax_acc.set_yticklabels(['$\emptyset$', 25, 50, 75, 100])
    ax_acc.tick_params(labelsize=16)

    # ---------- axes: right (classified) ----------
    ax_cls.set_xlabel("Years since EVAR", fontsize=20, labelpad=10)
    ax_cls.set_ylabel("Classified (%)", fontsize=20, labelpad=4)
    ax_cls.set_xlim(0, 5.5)
    ax_cls.set_ylim(-10, 110)
    ax_cls.set_xticks([0, 1, 2, 3, 4, 5])
    ax_cls.set_xticklabels(["EVAR", "1", "2", "3", "4", "5"])
    ax_cls.set_yticks([0, 25, 50, 75, 100])
    ax_cls.set_yticklabels([0, 25, 50, 75, 100])
    ax_cls.tick_params(labelsize=16)

    noise_handles = [
        Line2D([0],[0], color=noise_palette[n], lw=3, label=f"Spatial: {n*100:g}%")
        for n in unique_noises
    ]
    leg_noise = ax_cls.legend(
        handles=noise_handles,
        loc='upper left',
        bbox_to_anchor=(0.42, 0.42),
        frameon=False,
        fontsize=12
    )
    style_handles = []
    labels = {
        '0': 'Temporal: 1 year (optimal)',
        '1': 'Temporal: 0.5–1.5 years (ideal)',
        '2': 'Temporal: 1–3 years (reality)'
    }
    for k, st in interval_styles.items():
        h = Line2D([0],[0], color='black', lw=2, label=labels.get(k, k))
        if isinstance(st, tuple):
            h.set_dashes(st[1])
        else:
            h.set_linestyle(st)
        style_handles.append(h)
    leg_style = ax_cls.legend(
        handles=style_handles,
        loc='upper left',
        bbox_to_anchor=(0.28, 0.23),
        frameon=False,
        fontsize=12
    )
    ax_cls.add_artist(leg_noise)

    plt.tight_layout()
    plt.show()

def _compute_metrics(df_fold, low=0.1, high=0.9):
    df_valid = df_fold.dropna(subset=['p_c2', 'ground_truth'])
    n_total = len(df_valid)
    if n_total == 0:
        return np.nan, np.nan
    classified_mask = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)
    correct_mask = (
        (df_valid['ground_truth'] == 'Stable') & (df_valid['p_c2'] > high)
    ) | (
        (df_valid['ground_truth'] == 'Regr') & (df_valid['p_c2'] < low)
    )
    n_classified = classified_mask.sum()
    n_correct    = (classified_mask & correct_mask).sum()
    acc = n_correct / n_classified if n_classified > 0 else np.nan
    classified_frac = n_classified / n_total
    return acc, classified_frac