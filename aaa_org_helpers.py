import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from numpy.polynomial.polynomial import polyfit, polyval
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

rcParams['mathtext.fontset']='cm'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

colors = {'Regr': '#1f77b4', 'Stable': '#d62728'}

def normalize(data, xvar, yvar):
    normx = data[xvar].mean()
    normy = data[yvar].mean()
    data2 = data.copy()
    data2[xvar + '_Norm'] = data2[xvar]/normx
    data2[yvar + '_Norm'] = data2[yvar]/normy
    return data2, normx, normy

def compress_scales_NonPath(df):
    numeric_cols = df.select_dtypes(include='number').columns.difference(['Scan_ID'])
    agg_dict = {col: ['mean', 'std'] for col in numeric_cols}
    grouped = df.groupby('Scan_ID').agg(agg_dict)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()

def sample_via_poly(df, time_col='Years_mean', frame_col='frame', dt=0.25, N=30, order=3):
    out = []
    excl = {'Patient_ID', frame_col, time_col}
    feat_cols = df.select_dtypes('number').columns.difference(excl)
    for pid, g in df.groupby('Patient_ID'):
        g = g.sort_values(frame_col)
        t = g[time_col].values
        if len(t) == 0:
            continue
        t0, tN = t[0], t[-1]
        new_t = np.arange(t0, tN + 1e-8, dt)
        if len(new_t) > N:
            new_t = np.linspace(t0, tN, N)
        sampled = {'Patient_ID': pid, time_col: new_t}
        for feat in feat_cols:
            coefs = polyfit(t, g[feat].values, order)
            sampled[feat] = polyval(new_t, coefs)
        out.append(pd.DataFrame(sampled))
    if not out:
        return pd.DataFrame(columns=['Patient_ID', time_col] + list(feat_cols))
    return pd.concat(out, ignore_index=True)

def build_all_norm_compressed(path_to_data, xvar, yvar, path_sheet_names, fea_sheet_names=None):
    """
    Build normalized + compressed DataFrames for:
      - NonPath (sheet 'NonPath')
      - One or more Path cohorts (e.g. sheets 'UChicago', 'ErasmusU')
      - Optional FEA cohorts, each from its own sheet
        (e.g. 'UChicago_Regr_FEA', 'ErasmusU_Stable_FEA', ...)

    Returns
    -------
    dict
        Keys:
          - 'NonPath' : NonPath_norm_compressed DataFrame
          - Each Path sheet name (e.g. 'UChicago', 'ErasmusU')
          - Each FEA sheet name (e.g. 'UChicago_Regr_FEA', 'ErasmusU_Stable_FEA')
    """
    if isinstance(path_sheet_names, str):
        path_sheet_names = [path_sheet_names]
    if fea_sheet_names is not None and isinstance(fea_sheet_names, str):
        fea_sheet_names = [fea_sheet_names]

    os.chdir(path_to_data)
    excel_file = 'data.xlsx'  # change here if your file has a different name

    NonPath_xlsx = pd.read_excel(excel_file, sheet_name='NonPath')
    NonPath_xlsx[['Patient_ID', 'ScanNumber', 'Smoothing', 'Mesh']] = (
        NonPath_xlsx['ScanName'].str.split('_', expand=True)
    )

    combos_nonpath = [
        (0.5, 'M1open'),
        (0.5, 'M5open'),
        (0.5, 'M10open'),
        (1.0, 'M1open'),
        (1.0, 'M5open'),
        (1.0, 'M10open'),
        (5.0, 'M1open'),
        (5.0, 'M5open'),
        (5.0, 'M10open'),
    ]

    NonPath_norm_dfs = []
    norm_factors = {}
    for pf, mesh in combos_nonpath:
        subset = NonPath_xlsx[
            (NonPath_xlsx['Partition_Prefactor'] == pf) &
            (NonPath_xlsx['Mesh'] == mesh)
        ]
        if subset.empty:
            continue
        subset_norm, normx, normy = normalize(subset, xvar, yvar)
        NonPath_norm_dfs.append(subset_norm)
        norm_factors[(pf, mesh)] = (normx, normy)

    if NonPath_norm_dfs:
        NonPath_norm_concat = pd.concat(NonPath_norm_dfs, axis=0)
        NonPath_norm_concat.dropna(inplace=True)
    else:
        NonPath_norm_concat = pd.DataFrame()

    if not NonPath_norm_concat.empty:
        numeric_cols_non = (
            NonPath_norm_concat
            .select_dtypes(include='number')
            .columns
            .difference(['Scan_ID'])
        )
        agg_dict_non = {col: ['mean', 'std'] for col in numeric_cols_non}
        grouped_non = NonPath_norm_concat.groupby('Scan_ID').agg(agg_dict_non)
        grouped_non.columns = [f"{col}_{stat}" for col, stat in grouped_non.columns]
        NonPath_norm_compressed = grouped_non.reset_index()
    else:
        NonPath_norm_compressed = pd.DataFrame()

    def _compress_path_df(df_path_norm):
        if df_path_norm.empty:
            return df_path_norm

        numeric_cols = (
            df_path_norm
            .select_dtypes(include='number')
            .columns
            .difference(['Scan_ID'])
        )

        def most_common(series):
            return series.value_counts().idxmax()

        agg_dict = {col: ['mean', 'std'] for col in numeric_cols}
        if 'Op' in df_path_norm.columns:
            agg_dict['Op'] = most_common

        grouped = df_path_norm.groupby('Scan_ID').agg(agg_dict)

        new_cols = []
        for col, stat in grouped.columns:
            if col == 'Op':
                new_cols.append('Op')
            else:
                new_cols.append(f"{col}_{stat}")
        grouped.columns = new_cols

        return grouped.reset_index()

    def _compress_fea_df(df_fea_norm):
        if df_fea_norm.empty:
            return df_fea_norm

        numeric_cols_fea = (
            df_fea_norm
            .select_dtypes(include='number')
            .columns
            .difference(['Patient_ID', 'frame'])
        )
        agg_dict = {col: ['mean', 'std'] for col in numeric_cols_fea}
        grouped = df_fea_norm.groupby(['Patient_ID', 'frame']).agg(agg_dict)
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        return grouped.reset_index()

    path_results = {}

    for sheet_name in path_sheet_names:
        df_path = pd.read_excel(excel_file, sheet_name=sheet_name)
        df_path[['Patient_ID', 'ScanNumber', 'Smoothing', 'Mesh']] = (
            df_path['ScanName'].str.split('_', expand=True)
        )

        path_norm_dfs = []

        for pf, mesh in combos_nonpath:
            subset = df_path[
                (df_path['Partition_Prefactor'] == pf) &
                (df_path['Mesh'] == mesh)
            ]
            if subset.empty or (pf, mesh) not in norm_factors:
                continue

            normx, normy = norm_factors[(pf, mesh)]
            subset_norm = subset.copy()
            subset_norm[xvar + '_Norm'] = subset_norm[xvar] / normx
            subset_norm[yvar + '_Norm'] = subset_norm[yvar] / normy
            path_norm_dfs.append(subset_norm)

        if path_norm_dfs:
            path_norm_concat = pd.concat(path_norm_dfs, axis=0)
            path_norm_concat.dropna(inplace=True)
        else:
            path_norm_concat = pd.DataFrame()

        path_results[sheet_name] = _compress_path_df(path_norm_concat)

    fea_results = {}
    if fea_sheet_names is not None and len(fea_sheet_names) > 0:
        mesh_map = {'M1': 'M1open', 'M5': 'M5open', 'M10': 'M10open'}
        combos_fea = [
            (0.5, 'M1'),
            (0.5, 'M5'),
            (0.5, 'M10'),
            (1.0, 'M1'),
            (1.0, 'M5'),
            (1.0, 'M10'),
            (5.0, 'M1'),
            (5.0, 'M5'),
            (5.0, 'M10'),
        ]

        for fea_name in fea_sheet_names:
            df_fea = pd.read_excel(excel_file, sheet_name=fea_name)

            df_fea[['Patient_ID', 'ScanNumber', 'Smoothing',
                    'mapped', 'growth', 'frame', 'Mesh']] = (
                df_fea['ScanName'].str.split('_', expand=True)
            )

            fea_norm_dfs = []

            for pf, mesh_fea in combos_fea:
                subset = df_fea[
                    (df_fea['Partition_Prefactor'] == pf) &
                    (df_fea['Mesh'] == mesh_fea)
                ]
                if subset.empty:
                    continue

                mesh_non = mesh_map.get(mesh_fea)
                key_non = (pf, mesh_non)
                if key_non not in norm_factors:
                    continue  # no matching NonPath norms

                normx, normy = norm_factors[key_non]
                subset_norm = subset.copy()
                subset_norm[xvar + '_Norm'] = subset_norm[xvar] / normx
                subset_norm[yvar + '_Norm'] = subset_norm[yvar] / normy
                fea_norm_dfs.append(subset_norm)

            if fea_norm_dfs:
                fea_norm_concat = pd.concat(fea_norm_dfs, axis=0)
                fea_norm_concat.dropna(inplace=True)
            else:
                fea_norm_concat = pd.DataFrame()

            fea_results[fea_name] = _compress_fea_df(fea_norm_concat)

    out = {'NonPath': NonPath_norm_compressed}
    out.update(path_results)
    out.update(fea_results)
    return out

def sample_fea_results_via_poly(
    results,
    fea_keys=None,
    time_col='Years_mean',
    dt=0.25,
    N=30,
    order=3,
):
    """
    Take the `results` dict from build_all_norm_compressed, run sample_via_poly
    on all FEA entries, and return a new dict with downsampled FEA DataFrames.

    Parameters
    ----------
    results : dict
        Output from build_all_norm_compressed.
    fea_keys : list of str or None
        Which keys in `results` correspond to FEA sheets. If None, any key
        containing 'FEA' will be treated as FEA.
    time_col : str
        Name of the time column in the FEA DataFrames (default: 'Years_mean_mean').
    frame_col : str
        Name of the frame column (default: 'frame').
    dt : float
        Target time spacing for the first sampling attempt.
    N : int
        Max number of samples per Patient_ID (fallback if dt gives too many).
    order : int
        Polynomial order for the fit.

    Returns
    -------
    dict
        {fea_key: downsampled_df} for each FEA key.
    """
    # Infer FEA keys if not explicitly given
    if fea_keys is None:
        fea_keys = [k for k in results.keys() if 'FEA' in k]
    else:
        if isinstance(fea_keys, str):
            fea_keys = [fea_keys]

    downsampled = {}
    total_removed = 0

    for key in fea_keys:
        df = results.get(key)
        if df is None:
            continue

        orig_len = len(df)

        if orig_len == 0:
            downsampled[key] = df.copy()
            continue

        # run the sampling on this FEA sheet
        ds = sample_via_poly(
            df,
            time_col=time_col,
            frame_col=frame_col,
            dt=dt,
            N=N,
            order=order,
        )

        new_len = len(ds)
        removed = orig_len - new_len
        total_removed += removed

        print(f"[FEA sampling] {key}: {orig_len} -> {new_len} rows "
              f"({removed} removed)")

        downsampled[key] = ds

    print(f"[FEA sampling] Total points removed across all FEA sheets: {total_removed}")
    return downsampled

def avg_yr_interval(df):
    df = df.copy()
    df['Base_ID'] = df['Scan_ID'].str.rsplit('_', n=1).str[0]
    df = df.sort_values(['Base_ID', 'Years_mean'])
    df['delta_years'] = df.groupby('Base_ID')['Years_mean'].diff()
    avg_interval = df['delta_years'].dropna().mean()
    print("Average interval between consecutive scans (years):", avg_interval)

def compute_finite_differences(
    df,
    mode='both',
    id_col='Patient_ID',
    time_col='Years_mean',
    *,
    verbose=False
):
    """
    Compute finite differences of all numeric columns vs time_col,
    optionally verbose about which rows/patients are removed by
    cleaning criteria.
    """

    def clean_finite_differences(df_in, time_step=0.1, Kvals=(6.0, -2.5)):
        """
        Apply cleaning rules:

          - Δt > time_step
          - dIntGaussian_Fluct_Norm_mean/dYears_mean < Kvals[0]
          - dIntGaussian_Fluct_Norm_mean/dYears_mean > Kvals[1]

        If verbose=True, print which patients/rows are removed
        and which criteria they failed.
        """
        delta_col = f'Delta_{time_col}'
        deriv_col = 'dIntGaussian_Fluct_Norm_mean/dYears_mean'

        df_local = df_in.copy()

        # Boolean "fail" masks for clarity
        fail_dt    = df_local[delta_col] <= time_step
        fail_upper = df_local[deriv_col] >= Kvals[0]
        fail_lower = df_local[deriv_col] <= Kvals[1]

        # Keep rows that pass ALL criteria
        ok_mask = ~(fail_dt | fail_upper | fail_lower)

        if verbose:
            removed = df_local[~ok_mask].copy()
            if len(removed):
                removed['fail_dt']    = fail_dt[~ok_mask]
                removed['fail_upper'] = fail_upper[~ok_mask]
                removed['fail_lower'] = fail_lower[~ok_mask]

                total_removed = len(removed)
                print(f"[clean_finite_differences] Total rows removed: {total_removed}")

                for pid, sub in removed.groupby(id_col):
                    issues = []
                    if sub['fail_dt'].any():
                        issues.append(
                            f"Δ{time_col} <= {time_step} (n={sub['fail_dt'].sum()})"
                        )
                    if sub['fail_upper'].any():
                        issues.append(
                            f"{deriv_col} >= {Kvals[0]} (n={sub['fail_upper'].sum()})"
                        )
                    if sub['fail_lower'].any():
                        issues.append(
                            f"{deriv_col} <= {Kvals[1]} (n={sub['fail_lower'].sum()})"
                        )
                    issue_str = "; ".join(issues) if issues else "unknown criteria"
                    print(
                        f"  Patient {pid}: removed {len(sub)} rows due to {issue_str}"
                    )

                kept = df_local[ok_mask]
                print(
                    f"[clean_finite_differences] Rows kept after cleaning: "
                    f"{len(kept)} (from {len(df_local)})"
                )

        return df_local[ok_mask]

    # ---------- finite difference computation ----------
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != time_col
    ]

    results = []
    for pid, group in df.groupby(id_col):
        g = group.sort_values(by=time_col)
        times = g[time_col].values
        vals  = g[numeric_cols].values

        if len(times) < 2:
            if verbose:
                print(f"[compute_finite_differences] Patient {pid} skipped (only {len(times)} time point).")
            continue

        dt = times[1:] - times[:-1]
        derivs = (vals[1:] - vals[:-1]) / dt[:, None]

        for i in range(len(dt)):
            t0, t1 = times[i], times[i+1]
            v0, v1 = vals[i], vals[i+1]
            d      = derivs[i]

            def make_row(origin_time, origin_vals):
                row = {id_col: pid, time_col: origin_time}
                for j, col in enumerate(numeric_cols):
                    row[col] = origin_vals[j]
                    row[f'd{col}/d{time_col}'] = d[j]
                row[f'Delta_{time_col}'] = dt[i]
                return row

            if mode == 'preop':
                results.append(make_row(t0, v0))
            elif mode == 'postop':
                results.append(make_row(t1, v1))
            elif mode == 'midpoint':
                results.append(make_row((t0 + t1) / 2.0, (v0 + v1) / 2.0))
            elif mode == 'both':
                results.append(make_row(t0, v0))
                results.append(make_row(t1, v1))
            else:
                raise ValueError("Invalid mode: choose 'preop','postop','midpoint' or 'both'")

    out = pd.DataFrame(results)
    if out.empty:
        if verbose:
            print("[compute_finite_differences] No finite-difference rows generated.")
        return out

    out = out.sort_values([time_col]).reset_index(drop=True)
    return clean_finite_differences(out)

def report_patient_loss(orig_df, filt_df, label, id_col='Patient_ID'):
    """Print which patients vanished entirely after filtering."""
    orig_ids = set(orig_df[id_col].astype(str).unique())
    new_ids  = set(filt_df[id_col].astype(str).unique())
    lost_ids = sorted(orig_ids - new_ids)

    print(f"\n[{label}] Patients in original: {len(orig_ids)}")
    print(f"[{label}] Patients after filtering: {len(new_ids)}")
    print(f"[{label}] Patients completely removed: {len(lost_ids)}")

    if lost_ids:
        print(f"[{label}] Removed Patient_IDs: {', '.join(lost_ids)}")
    else:
        print(f"[{label}] No patients were completely removed.")

def summarize_AK(df, x_col='SurfaceArea_Norm_mean', k_col='IntGaussian_Fluct_Norm_mean'):
    """Return mean/std of A and K for a given dataframe."""
    A = df[x_col].to_numpy()
    K = df[k_col].to_numpy()
    return {
        'A0':    np.mean(A),
        'A0std': np.std(A, ddof=0),
        'K0':    np.mean(K),
        'K0std': np.std(K, ddof=0),
    }

def plot_dataset_overview(data_r, data_s, colors=colors):
    """
    Visualize overview stats comparing Regressed (data_r) vs Stable (data_s)
    """

    # Helper: compute per-patient scan count and mean interval
    def patient_stats(df):
        p_counts = df.groupby('Patient_ID').size()
        p_gaps = df.groupby('Patient_ID')['Years_mean'].apply(
            lambda x: np.diff(np.sort(x)).mean() if len(x) > 1 else np.nan
        )
        return p_counts, p_gaps

    # Compute stats
    counts_r, gaps_r = patient_stats(data_r)
    counts_s, gaps_s = patient_stats(data_s)

    n_patients_r = counts_r.shape[0]
    n_patients_s = counts_s.shape[0]

    mean_gaps_r = gaps_r.mean(skipna=True)
    mean_gaps_s = gaps_s.mean(skipna=True)

    # — Plot —
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: number of unique patients
    axes[0].bar(['Regr', 'Stable'], [n_patients_r, n_patients_s],
                color=[colors['Regr'], colors['Stable']])
    axes[0].set_ylabel('Unique Patients')
    axes[0].set_title('Number of Patients')

    # Panel 2: histogram of scans per patient
    axes[1].hist(counts_s, bins=np.arange(1, counts_s.max()+2)-0.5,
                 alpha=1.0, color=colors['Stable'], label='Stable')
    axes[1].hist(counts_r, bins=np.arange(1, counts_r.max()+2)-0.5,
                 alpha=1.0, color=colors['Regr'], label='Regr')
    axes[1].set_xlabel('Scans per Patient')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Scans per Patient')
    axes[1].legend()

    # Panel 3: histogram of average time between scans
    axes[2].hist(gaps_r.dropna(), bins=10, alpha=0.7,
                 color=colors['Regr'], label='Regr')
    axes[2].hist(gaps_s.dropna(), bins=10, alpha=0.7,
                 color=colors['Stable'], label='Stable')
    axes[2].set_xlabel('Mean Time Between Scans (years)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Mean Time Between Scans')
    axes[2].legend()

    fig.suptitle(
        f"Regressing: {n_patients_r} patients | Stable: {n_patients_s} patients\n"
        f"Mean time between scans: Regressing = {mean_gaps_r:.1f} yrs | Stable = {mean_gaps_s:.1f} yrs",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def plot_data_attrition(data_r, data_s, time_col='Years_mean', colors=colors):
    """
    Plot patient attrition curves.
    
    If split_subgroups=True, splits Regr & Stable by 'JK' and 'EU' in Patient_ID.
    """

    def attrition_curve(df, time_col):
        # Get each patient's all timepoints
        pid_time = df[['Patient_ID', time_col]].drop_duplicates()
        # For each unique time, count how many patients have >= that time
        times = np.sort(pid_time[time_col].unique())
        patient_remaining = []
        for t in times:
            # patients with at least one scan at t or later
            pids = pid_time.groupby('Patient_ID')[time_col].max()
            patient_remaining.append((pids >= t).sum())
        return times, np.array(patient_remaining)

    fig, ax = plt.subplots(figsize=(8,5))
    times_r, rem_r = attrition_curve(data_r, time_col)
    times_s, rem_s = attrition_curve(data_s, time_col)

    ax.step(times_r, rem_r, where='post', label='Regressing', lw=2, color=colors['Regr'])
    ax.step(times_s, rem_s, where='post', label='Stable', lw=2, color=colors['Stable'])

    ax.set_xlabel(f'Time (years)', fontsize=18)
    ax.set_ylabel('Patients Remaining', fontsize=18)
    ax.set_title('Data Attrition Over Time', fontsize=18)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_patient_barcode(
    data_r, data_s, time_col='Years_mean', id_col='Patient_ID', scan_id_col='Scan_ID',
    *, dpi=600, save_path=None, transparent=False
):
    df_r = data_r.copy()
    df_r['group'] = 'Regr'
    df_s = data_s.copy()
    df_s['group'] = 'Stable'
    all_data = pd.concat([df_r, df_s], ignore_index=True)

    all_data['region'] = all_data[id_col].apply(
        lambda x: 'JK' if 'JK' in str(x) else ('EU' if 'EU' in str(x) else 'OTHER')
    )
    group_order = [
        ('EU', 'Stable'),
        ('JK', 'Stable'),
        ('EU', 'Regr'),
        ('JK', 'Regr'),
    ]
    group_order2 = [
        ('Erasmus U.', 'Stable'),
        ('U. Chicago', 'Stable'),
        ('Erasmus U.', 'Regressed'),
        ('U. Chicago', 'Regressed'),
    ]
    ordered_pids = []
    group_indices = []
    for region, group in group_order:
        group_pids = all_data[(all_data['region'] == region) & (all_data['group'] == group)][id_col].unique()
        group_indices.append((len(ordered_pids), len(ordered_pids) + len(group_pids)))
        ordered_pids.extend(group_pids)
    n_patients = len(ordered_pids)

    pid_to_y = {pid: i for i, pid in enumerate(ordered_pids)}

    color_map = {'Regr': '#1f77b4', 'Stable': '#d62728'}
    group_bg = ["#000000", "#ffffff", "#000000", "#ffffff"]

    # Create figure at the requested DPI
    fig, ax = plt.subplots(figsize=(5, max(1.8, n_patients * 0.075)), dpi=dpi)

    line_len = 0.14
    for idx, row in all_data.iterrows():
        pid = row[id_col]
        if pid not in pid_to_y:
            continue
        y = pid_to_y[pid]
        t = row[time_col]
        plot_t = t if t <= 5 else 5.05
        ax.plot([plot_t - line_len/2, plot_t + line_len/2], [y, y],
                color=color_map[row['group']], lw=2, ls='-', alpha=0.8)

    # Group shading and labels
    for i, ((start, end), (region, group), (local, outcome)) in enumerate(zip(group_indices, group_order, group_order2)):
        ax.axhspan(start-0.5, end-0.5, facecolor=group_bg[i], alpha=0.25, zorder=-10)
        ax.text(0.01, (start + end - 1)/2, f'{local}\n{outcome}',
                va='center', ha='left', fontsize=10, fontweight='bold', transform=ax.get_yaxis_transform())
        if i < len(group_indices) - 1:
            ax.axhline(end-0.5, color='k', linestyle='dashed', linewidth=1, alpha=0.7, zorder=5)

    ax.set_yticks([])
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['EVAR', 1, 2, 3, 4, '5+'], fontsize=12)
    ax.set_ylabel("Patients")
    ax.set_xlabel("Time [yrs]")
    for group, color in color_map.items():
        ax.plot([], [], color=color, ls='--', lw=2, label=group)
    ax.set_ylim(-0.5, n_patients-0.5)
    ax.set_xlim(-1.2, 5.15)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    plt.tight_layout()

    # Save at 600 dpi if a path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=transparent)

    return fig, ax

def single_patient_trajectory(
    df,
    highlight_ids,
    *,
    id_col='Scan_ID',               
    x_col='SurfaceArea_Norm_mean',
    y_col='IntGaussian_Fluct_Norm_mean',
    xerr_col='SurfaceArea_Norm_std',
    yerr_col='IntGaussian_Fluct_Norm_std',
    show_background=True,
    highlight_colors=None,
    order_col=None,
    xlim=None,
    ylim=None,
    figsize=(5.5, 5.5),
    dpi=300,
    save=None
):
    d = df.copy()
    d[id_col] = d[id_col].astype(str)

    # --- substring-based highlighting ---
    patterns = [str(x) for x in np.atleast_1d(highlight_ids)]
    if len(patterns):
        regex = "|".join(map(re.escape, patterns))
        d['is_highlight'] = d[id_col].str.contains(regex)
    else:
        d['is_highlight'] = False

    bg = d[~d['is_highlight']]
    fg = d[d['is_highlight']]

    # Pick subset of points per highlighted ID
    selected = []
    for pid, g in fg.groupby(id_col, sort=False):
        if order_col is not None and order_col in g.columns:
            g = g.sort_values(by=order_col, kind='mergesort')
        else:
            g = g.sort_index()

        n = len(g)
        if n == 0:
            continue

        # indices for first, last, and two in between (or fewer if n<4)
        take = np.floor(np.linspace(0, max(n - 1, 0), min(4, n))).astype(int)
        g_sel = g.iloc[take]
        selected.append(g_sel)

    fg4 = pd.concat(selected) if len(selected) else fg.iloc[0:0]

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # Background points
    if show_background and len(bg):
        ax.errorbar(
            bg[x_col].to_numpy(), bg[y_col].to_numpy(),
            xerr=bg[xerr_col].to_numpy(), yerr=bg[yerr_col].to_numpy(),
            fmt='o', lw=0.8,
            color='k', ecolor='0.85', elinewidth=0.6,
            alpha=0.6, zorder=1
        )

    # Highlighted points
    if len(fg4):
        if highlight_colors is None:
            # default: same color for all highlighted IDs
            def get_color(_pid): return 'b'
        else:
            def get_color(_pid): return highlight_colors.get(_pid, 'b')

        for pid, sub in fg4.groupby(id_col, sort=False):
            c = get_color(pid)
            ax.errorbar(
                sub[x_col].to_numpy(), sub[y_col].to_numpy(),
                xerr=sub[xerr_col].to_numpy(), yerr=sub[yerr_col].to_numpy(),
                fmt='o', ms=15, lw=1.5, capsize=5,
                color=c, ecolor=c, elinewidth=1.0,
                alpha=1.0, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8
            )

    ax.set_xlabel('$\\widetilde{A}$', fontsize=24, labelpad=10)
    ax.set_ylabel('$\\widetilde{\\delta K}$', fontsize=24, labelpad=10)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis='both', which='both',
                   labelsize=14, direction='in',
                   length=5, width=1.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.5, alpha=0.5)

    if save:
        fig.savefig(save, bbox_inches='tight')

    return fig, ax