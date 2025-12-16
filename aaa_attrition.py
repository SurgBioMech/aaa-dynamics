import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import logsumexp
from matplotlib.lines import Line2D
from aaa_org_helpers import *
from aaa_zsindy import *
from aaa_bayesclf import *
from matplotlib import rcParams
rcParams['mathtext.fontset']='cm'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def _attrition_arrays(df, id_col="Patient_ID", time_col="Years_mean", min_step=True):
    """Return (times, counts_remaining, proportions_remaining, n_patients)."""
    if df.empty:
        return np.array([]), np.array([]), np.array([]), 0

    # last time per patient
    pid_time = df[[id_col, time_col]].drop_duplicates()
    last_time = (pid_time.groupby(id_col, as_index=False)[time_col]
                        .max()[time_col].to_numpy())
    n_patients = last_time.size
    if n_patients == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # evaluation times = unique scan times
    times = np.sort(pid_time[time_col].unique())

    # patients remaining at each time (last_time >= t)
    rem_counts = (last_time[:, None] >= times[None, :]).sum(axis=0)

    # prepend a starting point so step(where='post') begins at the full cohort
    if min_step:
        times = np.r_[times[0], times]
        rem_counts = np.r_[n_patients, rem_counts]

    rem_props = rem_counts / n_patients
    return times, rem_counts, rem_props, n_patients

def plot_data_attrition_2groups(data_1, data_2,
                                cat_1, cat_2, color_1, color_2,
                                id_col="Patient_ID", time_col="Years_mean"):
    
    colors = {cat_1: color_1, cat_2: color_2}
    cohorts = {
        cat_1:     data_1,
        cat_2:     data_2,
    }

    # compute once per cohort
    curves = {}
    for name, df in cohorts.items():
        t, c, p, n = _attrition_arrays(df, id_col=id_col, time_col=time_col)
        curves[name] = (t, c, p, n)

    # --- Figure 1: raw counts ---
    fig_counts, axc = plt.subplots(figsize=(10, 6))
    for name, (t, c, _, n) in curves.items():
        if t.size:
            axc.step(t, c, where='post', lw=2, label=f"{name} (N={n})", color=colors[name])
    axc.set_xlabel(f"Time ({time_col})")
    axc.set_ylabel("Patients remaining (count)")
    axc.set_title("Data Attrition Over Time — Counts")
    axc.grid(True, linestyle='--', alpha=0.4)
    axc.legend()
    plt.tight_layout()

    fig_props, axp = plt.subplots(figsize=(10, 6))
    prop_points = {}  # NEW: name -> (t, p)
    for name, (t, _, p, n) in curves.items():
        if t.size:
            axp.step(t, p, where='post', lw=2, label=f"{name} (N={n})", color=colors[name])
            prop_points[name] = (t, p)           # << capture (t, p)
    axp.set_xlabel(f"Time ({time_col})"); axp.set_ylabel("Patients remaining (proportion)")
    axp.set_ylim(0, 1.05)
    axp.set_title("Data Attrition Over Time — Proportions")
    axp.grid(True, linestyle='--', alpha=0.4); axp.legend(); plt.tight_layout()

    # return the same two figure handles PLUS the (t,p) for each curve
    return (fig_counts, axc), (fig_props, axp), prop_points

def static_bayes_class_att(
    df1_kde,
    df2_kde,
    df1,
    df2,
    t_min,
    t_max,
    prior,
    time_col='Years_mean',
    id_col='Patient_ID',
    prop_points=None,
    df1_label='Regressing',
    df2_label='Stable',
    eval_time=None,
    attrition_mode="dropout",
    random_state=None,
):
    """
    Static Bayesian classifier at time t_max using **history**.
    For each patient:
      - Collect all scans with t_min < time_col <= t_max.
      - For each scan k with state x_k, treat x_k as an independent draw
        from the KDE at this evaluation time (t_max).
      - Form total likelihood: product over scans (in log-space).
      - Combine with prior p(c) to get p(c | {x_k}, {t_k}).

    attrition_mode:
        "dropout":   randomly drop patients according to prop_points at t_max.
        "none":      ignore attrition info.
    """

    # Helper: select all scans in window (t_min, t_max]
    def select_window(df):
        return df[(df[time_col] > t_min) & (df[time_col] <= t_max)].copy()

    df1_sel = select_window(df1)
    df1_sel['ground_truth'] = df1_label

    df2_sel = select_window(df2)
    df2_sel['ground_truth'] = df2_label

    all_rows = pd.concat([df1_sel, df2_sel], ignore_index=True)

    # baseline effective prior at this evaluation time
    prior_t = float(prior)

    # No data in this time window at all
    if all_rows.empty:
        empty = pd.DataFrame(columns=[id_col, 'p_c2', 'ground_truth', 'n_scans'])
        return empty, prior_t

    # If we're in dropout mode: randomly remove patients according to prop_points
    if attrition_mode == "dropout" and prop_points is not None:
        t_eval = t_max if eval_time is None else float(eval_time)

        all_rows = _apply_random_attrition(
            all_rows,
            t_eval=t_eval,
            prop_points=prop_points,
            id_col=id_col,
            class_col="ground_truth",
            random_state=random_state,
            exact=True,
        )

        # If everyone dropped out, still return an empty df + attrition-adjusted prior
        if all_rows.empty:
            t1, p1 = prop_points[df1_label]
            t2, p2 = prop_points[df2_label]
            s1 = float(np.interp(t_eval, np.asarray(t1, float), np.asarray(p1, float)))
            s2 = float(np.interp(t_eval, np.asarray(t2, float), np.asarray(p2, float)))
            s1 = max(s1, 1e-12)
            s2 = max(s2, 1e-12)
            prior_t = (prior * s2) / ((1.0 - prior) * s1 + prior * s2)
            empty = pd.DataFrame(columns=[id_col, 'p_c2', 'ground_truth', 'n_scans'])
            return empty, prior_t

        # survival-adjusted prior
        t1, p1 = prop_points[df1_label]
        t2, p2 = prop_points[df2_label]
        s1 = float(np.interp(t_eval, np.asarray(t1, float), np.asarray(p1, float)))
        s2 = float(np.interp(t_eval, np.asarray(t2, float), np.asarray(p2, float)))
        s1 = max(s1, 1e-12)
        s2 = max(s2, 1e-12)

        prior_t = (prior * s2) / ((1.0 - prior) * s1 + prior * s2)

    # If KDEs are missing, everyone gets prior_t
    if (df1_kde is None) or (df2_kde is None):
        base = (all_rows[[id_col, 'ground_truth']]
                .drop_duplicates(subset=[id_col])
                .reset_index(drop=True))
        base['p_c2'] = prior_t
        base['n_scans'] = 0
        return base[[id_col, 'p_c2', 'ground_truth', 'n_scans']], prior_t

    log_prior_df1 = np.log(1.0 - prior_t)
    log_prior_df2 = np.log(prior_t)

    out_rows = []

    # One posterior per patient
    for pid, grp in all_rows.groupby(id_col):

        X = grp[['SurfaceArea_Norm_mean',
                 'IntGaussian_Fluct_Norm_mean']].to_numpy()

        n_scans = X.shape[0]

        if n_scans == 0:
            out_rows.append({
                id_col: pid,
                'p_c2': prior_t,
                'ground_truth': grp['ground_truth'].iloc[0],
                'n_scans': 0,
            })
            continue

        p_df1 = np.clip([df1_kde.pdf(x) for x in X], 1e-12, None)
        p_df2 = np.clip([df2_kde.pdf(x) for x in X], 1e-12, None)

        log_like_df1 = np.sum(np.log(p_df1))
        log_like_df2 = np.sum(np.log(p_df2))

        lv_df1 = log_like_df1 + log_prior_df1
        lv_df2 = log_like_df2 + log_prior_df2
        lv = np.array([lv_df1, lv_df2])

        lv -= logsumexp(lv)
        post = np.exp(lv)

        out_rows.append({
            id_col: pid,
            'p_c2': post[1],  # probability of df2_label
            'ground_truth': grp['ground_truth'].iloc[0],
            'n_scans': n_scans,
        })

    return pd.DataFrame(out_rows), prior_t

def static_classifier_plot(
    df0_static, 
    kde_df1_dict, kde_df2_dict, 
    data_df1_static, data_df2_static, 
    timepoints = [1, 2, 3],
    labels = ['Index\nTEVAR', '1 Yr.', '2 Yrs.', '3 Yrs.'],
    cat1 = "Regressing", color1 = "darkred",
    cat2 = "Stable",    color2 = "gray",
    prop_points=None,
    attrition_mode="dropout",   # "weights", "dropout", "none"
    random_state=None, 
    prior=0.5,
    dpi=100
):
    """
    Wrapper to run static_bayes_class_att at several timepoints
    and plot the jittered posterior points.

    Figure aesthetic matches jitter_plot / MAIN_FIG2:
      - Colored classification bands with vertical decision lines at (0.1, 0.9).
      - Jittered y positions in [0,1].
      - Right-hand margin text with classification stats and priors.
    """

    # --- helper for Times New Roman ticks, like jitter_plot ---
    def _force_tnr_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)
        for lab in ax.get_yticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)

    color_map = {cat1: color1, cat2: color2}
    low, high = 0.10, 0.90
    alpha_points = 0.50

    # dfs_static[0] is the TEVAR baseline (no classifier yet)
    dfs_static = [df0_static]
    priors = [prior]

    for i, t in enumerate(timepoints):
        df_t, prior_used = static_bayes_class_att(
            df1_kde      = kde_df1_dict[t],
            df2_kde      = kde_df2_dict[t],
            df1          = data_df1_static,
            df2          = data_df2_static,
            df1_label    = cat1,
            df2_label    = cat2,
            t_min        = -91/365,   # include peri-op anchored derivatives
            t_max        = t,
            prior        = prior,
            prop_points  = prop_points,
            attrition_mode = attrition_mode,
            random_state   = None if random_state is None else random_state + i,
        )
        dfs_static.append(df_t)
        priors.append(prior_used)

    # For reference counts (scans up to each time) – still used for debug print
    result_static = pd.concat(
        [data_df1_static, data_df2_static],
        ignore_index=True
    )
    results_static = [pd.DataFrame({'Patient_ID': []})]
    for tp in timepoints:
        results_static.append(
            result_static[result_static['Years_mean'] <= tp]
        )

    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True, facecolor='white', dpi=dpi)

    handles = [
        Line2D([0], [0], marker='o', color=color1, linestyle='None',
               markersize=8, label=cat1),
        Line2D([0], [0], marker='o', color=color2, linestyle='None',
               markersize=8, label=cat2)
    ]
    #axes[0].legend(handles=handles, loc='upper left', fontsize=10)

    for idx, (ax, df_t, lab, r) in enumerate(zip(axes, dfs_static, labels, results_static)):
        df_t = df_t.copy()

        # Jitter in [0,1]
        if not df_t.empty:
            y_jit = np.random.rand(len(df_t))
        else:
            y_jit = np.array([])

        # Scatter points with same layering logic as jitter_plot
        if not df_t.empty and ('p_c2' in df_t.columns):
            gt = df_t['ground_truth']

            # front/back ordering:
            # - "Stable" and "Regressing" map to cat2 / cat1
            # - splitting by left/right of 0.5 to match other jitter panels
            mask_cat1 = (gt == cat1)
            mask_cat2 = (gt == cat2)

            blue_left   = mask_cat1 & (df_t['p_c2'] < 0.5)
            blue_right  = mask_cat1 & (df_t['p_c2'] >= 0.5)
            red_left    = mask_cat2 & (df_t['p_c2'] < 0.5)
            red_right   = mask_cat2 & (df_t['p_c2'] >= 0.5)
            grey        = (df_t['p_c2'] == 0.5)

            # Stable left (back), Regr left (front)
            ax.scatter(
                df_t.loc[red_left, 'p_c2'],
                y_jit[red_left],
                c=color_map[cat2], alpha=alpha_points, s=30, zorder=5
            )
            ax.scatter(
                df_t.loc[blue_left, 'p_c2'],
                y_jit[blue_left],
                c=color_map[cat1], alpha=alpha_points, s=30, zorder=10
            )
            # Regr right (back), Stable right (front)
            ax.scatter(
                df_t.loc[blue_right, 'p_c2'],
                y_jit[blue_right],
                c=color_map[cat1], alpha=alpha_points, s=30, zorder=5
            )
            ax.scatter(
                df_t.loc[red_right, 'p_c2'],
                y_jit[red_right],
                c=color_map[cat2], alpha=alpha_points, s=30, zorder=10
            )
            # Equivocal
            ax.scatter(
                df_t.loc[grey, 'p_c2'],
                y_jit[grey],
                c='grey', alpha=1.0, s=30, zorder=15
            )
            # Prior Line
            ax.axvline(
                priors[idx], color='black', linestyle='--', lw=1.5, alpha=1, zorder=20
            )

            if idx == 0:
                ax.text(
                    0.64, 0.75,
                    "Prior Line",
                    rotation=0,
                    fontsize=10,
                    fontname="Times New Roman",
                    color='black',
                    zorder=25
                )

        ax.set_yticks([])
        ax.set_ylabel(
            lab,
            rotation=0,
            labelpad=40,
            va='center',
            fontsize=16,
            fontname="Times New Roman"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_facecolor('white')

        # Classification bands and decision lines
        ax.axvline(low,  color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        ax.axvline(high, color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle(
            (0.0, y0), low, y1 - y0,
            facecolor=color_map[cat1], alpha=0.3, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (low, y0), high - low, y1 - y0,
            facecolor="#7e7e7e", alpha=0.45, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (high, y0), 1.0 - high, y1 - y0,
            facecolor=color_map[cat2], alpha=0.3, lw=0, zorder=0
        ))

        # -------- Metrics for right-hand margin text --------
        if (not df_t.empty) and ('p_c2' in df_t.columns):
            df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
            n_total = len(df_valid)

            if n_total > 0:
                mask_class = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)
                n_class = mask_class.sum()
                class_pct = 100.0 * n_class / n_total if n_total > 0 else np.nan

                # predicted label by 0.5 cutoff
                pred_label = np.where(df_valid['p_c2'] >= 0.5, cat2, cat1)
                correct = (pred_label == df_valid['ground_truth']) & mask_class
                n_correct = correct.sum()
                acc_pct = 100.0 * n_correct / n_class if n_class > 0 else np.nan
            else:
                class_pct = np.nan
                acc_pct = np.nan

            n_regr   = (df_valid['ground_truth'] == cat1).sum()
            n_stable = (df_valid['ground_truth'] == cat2).sum()
            n_synth  = df_valid['Patient_ID'].nunique()

        else:
            # e.g., TEVAR baseline row
            class_pct = np.nan
            acc_pct   = np.nan
            n_regr    = 0
            n_stable  = 0
            n_synth   = 0

        # priors[idx] is p(cat2) (Stable)
        prior_stable    = float(priors[idx])
        prior_regressing = 1.0 - prior_stable

        class_str = "Classified: " + (f"{class_pct:.1f}%" if not np.isnan(class_pct) else "n/a")
        acc_str   = "Accuracy: "   + (f"{acc_pct:.1f}%"   if not np.isnan(acc_pct)   else "n/a")

        info_text = (
            f"{class_str}\n"
            f"{acc_str}\n"
            f"Synthetic Patients: {n_synth}\n"
            f"Regressing Patients: {n_regr}\n"
            f"Stable Patients: {n_stable}\n"
            f"Regressing Prior: {prior_regressing:.2f}\n"
            f"Stable Prior: {prior_stable:.2f}"
        )

        # right-hand margin text, no bbox
        ax.text(
            1.05, 0.5,
            info_text,
            transform=ax.transAxes,
            ha='left', va='center',
            fontsize=10,
            fontname="Times New Roman"
        )

        # Only bottom axis gets x-ticks/label
        if ax is not axes[-1]:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_xlabel(
                r'$p(Stable\ Statics \mid Anatomic\ Derivatives)$',
                fontsize=12,
                fontname="Times New Roman"
            )

        _force_tnr_ticks(ax)

    plt.tight_layout(rect=[0, 0, 0.80, 0.96])
    plt.show()

    # debug print of patient counts per row (unchanged)
    for lab, df_t in zip(labels, dfs_static):
        if 'Patient_ID' in df_t.columns:
            print(lab, " n patients in classifier output:", df_t['Patient_ID'].nunique())
        else:
            print(lab, " n patients in classifier output: 0")

    return dfs_static, results_static

def _apply_random_attrition(
    df,
    t_eval,
    prop_points,
    id_col="Patient_ID",
    class_col="ground_truth",
    random_state=None,
    exact=True,
):
    """
    Randomly drop patients at evaluation time t_eval according to
    cohort-specific survival curves in prop_points.

    prop_points: dict {label -> (t_array, p_array)}, where p(t) is
                 the FRACTION of patients remaining at time t (0-1).
    """
    if prop_points is None or len(df) == 0:
        return df

    rng = np.random.default_rng(random_state)
    out_frames = []

    # one cohort at a time
    for label, grp in df.groupby(class_col):
        t_attr, p_attr = prop_points.get(
            label,
            (np.array([0.0], float), np.array([1.0], float))
        )
        t_attr = np.asarray(t_attr, float)
        p_attr = np.asarray(p_attr, float)

        # interpolated survival fraction for THIS cohort at t_eval
        surv_frac = float(np.interp(float(t_eval), t_attr, p_attr))
        surv_frac = float(np.clip(surv_frac, 0.0, 1.0))

        # which patients belong to that cohort
        pids = grp[id_col].unique()
        n0   = len(pids)

        if n0 == 0 or surv_frac <= 0:
            continue

        # expected survivors ~ surv_frac * n0
        k = int(round(surv_frac * n0))
        k = max(0, min(k, n0))

        if k == n0:
            chosen_pids = pids
        elif k == 0:
            chosen_pids = []
        else:
            chosen_pids = rng.choice(pids, size=k, replace=False)

        out_frames.append(grp[grp[id_col].isin(chosen_pids)])

    if not out_frames:
        return df.iloc[0:0].copy()

    return pd.concat(out_frames, ignore_index=True)

def dynamic_bayes_class_att(
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
    static_kde_dicts=None,
    static_round_to=None,
    attrition_on=False,
    attrition_config=None,
    make_fig=False,
    acc_bounds=(0.1, 0.9),
    axis_title="p(Stable)",
    color_map=None,
    alpha_points=0.50,
):
    """
    Dynamic + (optional) Static Bayesian classifier over two classes (Regr, Stable).

    Returns
    -------
    df_out : DataFrame
        One row per patient with columns:
            [id_col, 'p_c2', 'ground_truth']
        where p_c2 = p(Stable | data).
    prior_c2 : float
        The effective prior probability for the second class (Stable) after
        any attrition / survival adjustment.

    Additionally (side effect, if make_fig=True):
        Produces a single-panel jitter plot with jitter_plot-like aesthetic
        and right-margin text with:
            Classified %:
            Accuracy %:
            Synthetic Patients:
            Regressing Patients:
            Stable Patients:
            Regressing Prior:
            Stable Prior:
    """

    # helper for TNR ticks
    def _force_tnr_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)
        for lab in ax.get_yticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)

    if color_map is None:
        color_map = {'Regr': '#1f77b4', 'Stable': '#d62728'}

    low, high = acc_bounds

    # ----- ground truth tagging -----
    df_r = regr_df.copy()
    df_r['ground_truth'] = 'Regr'

    df_s = stable_df.copy()
    df_s['ground_truth'] = 'Stable'

    all_rows = pd.concat([df_r, df_s], ignore_index=True)
    prior_c2 = float(prior)
    prior_c1 = 1.0 - prior_c2
    if attrition_on and (attrition_config is not None):
        prop_points   = attrition_config.get('prop_points', None)
        time_cutoff   = attrition_config.get('time_cutoff', t_max)
        apply_dropout = attrition_config.get('apply_dropout', True)
        adjust_prior  = attrition_config.get('adjust_prior', True)
        random_state  = attrition_config.get('random_state', None)

        # (a) optional patient-level dropout first (per cohort)
        if apply_dropout and (prop_points is not None):
            all_rows = _apply_random_attrition(
                all_rows,
                t_eval=time_cutoff,
                prop_points=prop_points,
                id_col=id_col,
                class_col="ground_truth",
                random_state=random_state,
                exact=True,
            )

        # (b) optional attrition-aware prior update
        if adjust_prior and (prop_points is not None):
            def _survival_for(cls_label):
                t_arr, p_arr = prop_points.get(
                    cls_label,
                    (np.array([0.0], float), np.array([1.0], float))
                )
                t_arr = np.asarray(t_arr, float)
                p_arr = np.asarray(p_arr, float)
                return float(np.interp(time_cutoff, t_arr, p_arr))

            s_regr   = max(_survival_for('Regr'),   1e-12)
            s_stable = max(_survival_for('Stable'), 1e-12)

            pri_vec = np.array([prior_c1, prior_c2], float)
            pri_vec /= pri_vec.sum()
            pi1, pi2 = pri_vec[0], pri_vec[1]

            prior_c2 = (pi2 * s_stable) / (pi1 * s_regr + pi2 * s_stable)
            prior_c1 = 1.0 - prior_c2

    log_prior_regr   = np.log(prior_c1)
    log_prior_stable = np.log(prior_c2)

    def select_window(g):
        # keep slight tolerance like before
        return g[(g[time_col] > t_min - 1/12) &
                 (g[time_col] <= t_max + 1/12)].copy()

    model_regr_dict, model_stable_dict = results
    two_models = [model_regr_dict, model_stable_dict]
    rho_list   = [model_regr_dict['rho'], model_stable_dict['rho']]

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
        t_avail = np.array([tt for tt, v in kde_by_time.items() if v is not None])

        for x_k, t_k in zip(X, t_list):
            if t_avail.size == 0:
                logliks.append(0.0)
                continue

            if round_to is not None:
                t_key = np.round(t_k / round_to) * round_to
                kde = kde_by_time.get(t_key, None)
                if kde is None:
                    idx = np.argmin(np.abs(t_avail - t_k))
                    kde = kde_by_time[t_avail[idx]]
            else:
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
            gt = grp_full['ground_truth'].iloc[0]
            out.append({
                id_col: pid,
                'p_c2': prior_c2,
                'ground_truth': gt
            })
            continue

        comp = grp[[
            'SurfaceArea_Norm_mean',
            'IntGaussian_Fluct_Norm_mean',
            id_col,
            time_col,
            'ground_truth'
        ]].copy()

        comp['A_pred'] = comp['SurfaceArea_Norm_mean']
        comp['K_pred'] = comp['IntGaussian_Fluct_Norm_mean']

        X_spatial = comp[['SurfaceArea_Norm_mean',
                          'IntGaussian_Fluct_Norm_mean']].to_numpy()
        t_vals = comp[time_col].to_numpy()

        xdot_true_df = compute_finite_differences(
            comp,
            verbose=verbose_fd
        )

        if xdot_true_df.empty:
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
        d_dim = xdot_true.shape[1]

        log_z = np.zeros(2)

        for j, mdl in enumerate(two_models):
            model_j = mdl['model']
            x0_j    = mdl['x0']
            rho_j   = rho_list[j]

            x_pred = model_j.simulate(x0_j, t_vals)
            comp['A_pred'], comp['K_pred'] = x_pred[:, 0], x_pred[:, 1]

            xdot_pred_df = compute_finite_differences(
                comp,
                verbose=False
            )

            xdot_pred = xdot_pred_df.loc[idx_eff, [
                'dA_pred/dYears_mean',
                'dK_pred/dYears_mean'
            ]].to_numpy()

            SSE = np.sum((xdot_true - xdot_pred) ** 2)

            log_norm  = - (d_dim * n_eff / 2.0) * np.log(2.0 * np.pi * (rho_j ** 2))
            log_resid = - SSE / (2.0 * (rho_j ** 2))
            log_dyn   = log_norm + log_resid

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

    df_out = pd.DataFrame(out)

    if make_fig:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.0), facecolor='white')

        df_t = df_out.copy().reset_index(drop=True)

        if not df_t.empty:
            y_jit = np.random.rand(len(df_t))

            gt = df_t['ground_truth']
            mask_cat1 = (gt == 'Regr')
            mask_cat2 = (gt == 'Stable')

            blue_left   = mask_cat1 & (df_t['p_c2'] < 0.5)
            blue_right  = mask_cat1 & (df_t['p_c2'] >= 0.5)
            red_left    = mask_cat2 & (df_t['p_c2'] < 0.5)
            red_right   = mask_cat2 & (df_t['p_c2'] >= 0.5)
            grey        = (df_t['p_c2'] == 0.5)

            ax.scatter(df_t.loc[red_left,  'p_c2'], y_jit[red_left],
                       c=color_map['Stable'], alpha=alpha_points, s=30, zorder=5)
            ax.scatter(df_t.loc[blue_left, 'p_c2'], y_jit[blue_left],
                       c=color_map['Regr'],   alpha=alpha_points, s=30, zorder=10)
            ax.scatter(df_t.loc[blue_right, 'p_c2'], y_jit[blue_right],
                       c=color_map['Regr'],   alpha=alpha_points, s=30, zorder=5)
            ax.scatter(df_t.loc[red_right,  'p_c2'], y_jit[red_right],
                       c=color_map['Stable'], alpha=alpha_points, s=30, zorder=10)
            ax.scatter(df_t.loc[grey, 'p_c2'], y_jit[grey],
                       c='grey', alpha=1.0, s=30, zorder=15)

            ax.set_yticks([])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_facecolor('white')

            # bands + decision lines
            ax.axvline(low,  color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
            ax.axvline(high, color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
            y0, y1 = ax.get_ylim()
            ax.add_patch(Rectangle(
                (0.0, y0), low, y1 - y0,
                facecolor=color_map['Regr'], alpha=0.3, lw=0, zorder=0
            ))
            ax.add_patch(Rectangle(
                (low, y0), high - low, y1 - y0,
                facecolor="#7e7e7e", alpha=0.45, lw=0, zorder=0
            ))
            ax.add_patch(Rectangle(
                (high, y0), 1.0 - high, y1 - y0,
                facecolor=color_map['Stable'], alpha=0.3, lw=0, zorder=0
            ))

            # metrics for right side
            df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
            n_total = len(df_valid)

            if n_total > 0:
                mask_class = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)
                n_class = mask_class.sum()
                class_pct = 100.0 * n_class / n_total if n_total > 0 else np.nan

                pred_label = np.where(df_valid['p_c2'] >= 0.5, 'Stable', 'Regr')
                correct = (pred_label == df_valid['ground_truth']) & mask_class
                n_correct = correct.sum()
                acc_pct = 100.0 * n_correct / n_class if n_class > 0 else np.nan
            else:
                class_pct = np.nan
                acc_pct   = np.nan

            n_regr   = (df_valid['ground_truth'] == 'Regr').sum()
            n_stable = (df_valid['ground_truth'] == 'Stable').sum()
            n_synth  = df_valid[id_col].nunique()

            class_str = "Classified: " + (f"{class_pct:.1f}%" if not np.isnan(class_pct) else "n/a")
            acc_str   = "Accuracy: "   + (f"{acc_pct:.1f}%"   if not np.isnan(acc_pct)   else "n/a")

            info_text = (
                f"{class_str}\n"
                f"{acc_str}\n"
                f"Synthetic Patients: {n_synth}\n"
                f"Regressing Patients: {n_regr}\n"
                f"Stable Patients: {n_stable}\n"
                f"Regressing Prior: {prior_c1:.2f}\n"
                f"Stable Prior: {prior_c2:.2f}"
            )

            ax.text(
                1.02, 0.5,
                info_text,
                transform=ax.transAxes,
                ha='left', va='center',
                fontsize=11,
                fontname="Times New Roman"
            )

        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(axis_title, fontsize=14, fontname="Times New Roman")
        _force_tnr_ticks(ax)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.show()

    return df_out, prior_c2

def dynamic_classifier_plot(
    df1_ens, df2_ens,
    results_all,
    data_r_trans,
    data_s_trans,
    kde_regr_dict_trans,
    kde_stable_dict_trans,
    x0, x0_std,
    best_results_df1_df2,
    mode,
    timepoints = [1, 2, 3],
    labels     = ['Index\nTEVAR', '1 Yr.', '2 Yrs.', '3 Yrs.'],
    cat1 = "Regressing", color1 = "darkred",
    cat2 = "Stable",    color2 = "gray",
    caps_label1 = "REGR", caps_label2 = "STABLE",
    prop_points   = None,
    attrition_mode = "dropout",
    random_state   = None,
    prior=0.5,
    dpi=100
):
    """
    Dynamic classifier plot, assuming df1_ens and df2_ens are *already*
    ensemble trajectories (the outputs regr_ens, stable_ens, grow_ens, etc.).

    best_results_df1_df2 is the pair-specific (mdl1, mdl2) from
    run_sindy_double_pipelines_noplot, keyed by (cA, cB) in best_results_by_pair.

    Figure aesthetic matches jitter_plot / static_classifier_plot:
      - Colored classification bands with vertical decision lines at (0.1, 0.9).
      - Jittered y positions in [0,1].
      - Right-hand margin text with:
            Classified %:
            Accuracy %:
            Synthetic Patients:
            Regressing Patients:
            Stable Patients:
            Regressing Prior:
            Stable Prior:
    """

    # small helper for TNR ticks
    def _force_tnr_ticks(ax):
        for lab in ax.get_xticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)
        for lab in ax.get_yticklabels():
            lab.set_fontname("Times New Roman")
            lab.set_fontsize(12)

    color_map = {cat1: color1, cat2: color2}
    low, high = 0.10, 0.90
    alpha_points = 0.50

    # ---- 1. Prepare ensemble DFs: Patient_ID and Scan_ID ----
    df1_ens = df1_ens.copy()
    df2_ens = df2_ens.copy()

    df1_ens['Patient_ID'] = caps_label1 + df1_ens['trial'].astype(str)
    df2_ens['Patient_ID'] = caps_label2 + df2_ens['trial'].astype(str)

    df1_ens = df1_ens.sort_values(['Patient_ID', 'Years_mean']).copy()
    df2_ens = df2_ens.sort_values(['Patient_ID', 'Years_mean']).copy()

    df1_ens['order']   = df1_ens.groupby('Patient_ID').cumcount() + 1
    df2_ens['order']   = df2_ens.groupby('Patient_ID').cumcount() + 1
    df1_ens['Scan_ID'] = df1_ens['Patient_ID'] + '_' + df1_ens['order'].astype(str)
    df2_ens['Scan_ID'] = df2_ens['Patient_ID'] + '_' + df2_ens['order'].astype(str)

    # ---- 2. Transform (downsample + noise) if desired ----
    df1_ens_transformed = stress_test_transformer(
        df1_ens,
        add_noise=False,
        fixed_interval=1.0,
        keep_other_cols=True,
        random_interval_range=None,
    )

    df2_ens_transformed = stress_test_transformer(
        df2_ens,
        add_noise=False,
        fixed_interval=1.0,
        keep_other_cols=True,
        random_interval_range=None,
    )

    data_1 = df1_ens_transformed
    data_2 = df2_ens_transformed

    # full combined data for scan counts per time
    result = pd.concat([data_1, data_2], ignore_index=True)

    # TEVAR baseline: one row per patient, p_c2=0.5, n_scans=0
    df0_1 = data_1.drop_duplicates('Patient_ID')[['Patient_ID']].assign(ground_truth=cat1)
    df0_2 = data_2.drop_duplicates('Patient_ID')[['Patient_ID']].assign(ground_truth=cat2)
    df0   = pd.concat([df0_1, df0_2], ignore_index=True).drop_duplicates('Patient_ID')
    df0['p_c2']    = 0.5
    df0['n_scans'] = 0

    dfs_dyn     = [df0]
    results_dyn = [pd.DataFrame({'Patient_ID': []})]
    priors = [prior]

    # ---- 3. Build attrition curves for dynamic_bayes_class_att ----
    if prop_points is not None:
        t_regr, p_regr = prop_points["Regr"]
        t_stable, p_stable = prop_points["Stable"]
    else:
        t_regr = p_regr = t_stable = p_stable = None

    # ---- 4. Posteriors at each timepoint ----
    for i, t in enumerate(timepoints):

        if prop_points is not None:
            attrition_config = {
                "prop_points": {
                    "Regr":   (t_regr,   p_regr),
                    "Stable": (t_stable, p_stable),
                },
                "time_cutoff": t,
                "apply_dropout": (attrition_mode == "dropout"),
                "adjust_prior": True,
                "random_state": None if random_state is None else random_state + i,
            }
        else:
            attrition_config = None

        # here we assume you have results_all, data_r_trans, data_s_trans, and
        # (kde_regr_dict_trans, kde_stable_dict_trans) in scope as before
        df_dyn_t, prior_used = dynamic_bayes_class_att(
            results        = results_all,
            regr_df        = data_r_trans,
            stable_df      = data_s_trans,
            t_min          = 0.0,
            t_max          = t,
            prior          = prior,
            id_col         = 'Patient_ID',
            time_col       = 'Years_mean',
            verbose_fd     = False,
            static_kde_dicts = (kde_regr_dict_trans, kde_stable_dict_trans),
            attrition_on   = (attrition_mode == "dropout") and (prop_points is not None),
            attrition_config = attrition_config,
        )
        dfs_dyn.append(df_dyn_t)
        results_dyn.append(result[result['Years_mean'] <= t])
        priors.append(prior_used)

    # ---- 5. Plotting: jitter style, right-margin text ----
    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True, facecolor='white', dpi=dpi)
    for idx, (ax, df_t, lab, r) in enumerate(zip(axes, dfs_dyn, labels, results_dyn)):
        df_t = df_t.copy()

        if not df_t.empty:
            y_jit = np.random.rand(len(df_t))
        else:
            y_jit = np.array([])

        # scatter with same layering as jitter_plot
        if (not df_t.empty) and ('p_c2' in df_t.columns):
            gt = df_t['ground_truth']
            mask_cat1 = (gt == cat1)
            mask_cat2 = (gt == cat2)

            blue_left   = mask_cat1 & (df_t['p_c2'] < 0.5)
            blue_right  = mask_cat1 & (df_t['p_c2'] >= 0.5)
            red_left    = mask_cat2 & (df_t['p_c2'] < 0.5)
            red_right   = mask_cat2 & (df_t['p_c2'] >= 0.5)
            grey        = (df_t['p_c2'] == 0.5)

            ax.scatter(df_t.loc[red_left, 'p_c2'], y_jit[red_left],
                       c=color_map[cat2], alpha=alpha_points, s=30, zorder=5)
            ax.scatter(df_t.loc[blue_left, 'p_c2'], y_jit[blue_left],
                       c=color_map[cat1], alpha=alpha_points, s=30, zorder=10)
            ax.scatter(df_t.loc[blue_right, 'p_c2'], y_jit[blue_right],
                       c=color_map[cat1], alpha=alpha_points, s=30, zorder=5)
            ax.scatter(df_t.loc[red_right, 'p_c2'], y_jit[red_right],
                       c=color_map[cat2], alpha=alpha_points, s=30, zorder=10)
            ax.scatter(df_t.loc[grey, 'p_c2'], y_jit[grey],
                       c='grey', alpha=1.0, s=30, zorder=15)
            # Prior Line
            ax.axvline(
                priors[idx], color='black', linestyle='--', lw=1.5, alpha=1, zorder=20
            )

            if idx == 0:
                ax.text(
                    0.64, 0.75,
                    "Prior Line",
                    rotation=0,
                    fontsize=10,
                    fontname="Times New Roman",
                    color='black',
                    zorder=25
                )

        ax.set_yticks([])
        ax.set_ylabel(
            lab,
            rotation=0,
            labelpad=40,
            va='center',
            fontsize=16,
            fontname="Times New Roman"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_facecolor('white')

        # classification bands + decision lines
        ax.axvline(low,  color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        ax.axvline(high, color='k', linestyle='-', lw=1.5, alpha=1, zorder=15)
        y0, y1 = ax.get_ylim()
        ax.add_patch(Rectangle(
            (0.0, y0), low, y1 - y0,
            facecolor=color_map[cat1], alpha=0.3, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (low, y0), high - low, y1 - y0,
            facecolor="#7e7e7e", alpha=0.45, lw=0, zorder=0
        ))
        ax.add_patch(Rectangle(
            (high, y0), 1.0 - high, y1 - y0,
            facecolor=color_map[cat2], alpha=0.3, lw=0, zorder=0
        ))

        # metrics for right margin
        if (not df_t.empty) and ('p_c2' in df_t.columns):
            df_valid = df_t.dropna(subset=['p_c2', 'ground_truth'])
            n_total = len(df_valid)

            if n_total > 0:
                mask_class = (df_valid['p_c2'] < low) | (df_valid['p_c2'] > high)
                n_class = mask_class.sum()
                class_pct = 100.0 * n_class / n_total if n_total > 0 else np.nan

                pred_label = np.where(df_valid['p_c2'] >= 0.5, cat2, cat1)
                correct = (pred_label == df_valid['ground_truth']) & mask_class
                n_correct = correct.sum()
                acc_pct = 100.0 * n_correct / n_class if n_class > 0 else np.nan
            else:
                class_pct = np.nan
                acc_pct   = np.nan

            n_regr   = (df_valid['ground_truth'] == cat1).sum()
            n_stable = (df_valid['ground_truth'] == cat2).sum()
            n_synth  = df_valid['Patient_ID'].nunique()
        else:
            class_pct = np.nan
            acc_pct   = np.nan
            n_regr    = 0
            n_stable  = 0
            n_synth   = 0

        prior_stable    = float(priors[idx])
        prior_regressing = 1.0 - prior_stable

        class_str = "Classified: " + (f"{class_pct:.1f}%" if not np.isnan(class_pct) else "n/a")
        acc_str   = "Accuracy: "   + (f"{acc_pct:.1f}%"   if not np.isnan(acc_pct)   else "n/a")

        info_text = (
            f"{class_str}\n"
            f"Accuracy: {acc_str.split(': ')[1]}\n"
            f"Synthetic Patients: {n_synth}\n"
            f"Regressing Patients: {n_regr}\n"
            f"Stable Patients: {n_stable}\n"
            f"Regressing Prior: {prior_regressing:.2f}\n"
            f"Stable Prior: {prior_stable:.2f}"
        )

        ax.text(
            1.05, 0.5,
            info_text,
            transform=ax.transAxes,
            ha='left', va='center',
            fontsize=10,
            fontname="Times New Roman"
        )

        # x-axis labels only on bottom
        if ax is not axes[-1]:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_xlabel(
                r'$p(Stable\ Dynamics \mid Anatomic\ Derivatives)$',
                fontsize=12,
                fontname="Times New Roman"
            )

        _force_tnr_ticks(ax)

    plt.tight_layout(rect=[0, 0, 0.80, 0.96])
    plt.show()

    return dfs_dyn, results_dyn