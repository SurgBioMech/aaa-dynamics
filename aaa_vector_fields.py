import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.interpolate import griddata
import matplotlib as mpl
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import gridspec
import seaborn as sns
from scipy.interpolate import RBFInterpolator
from matplotlib.patches import FancyArrowPatch
from sklearn.neighbors import NearestNeighbors
from collections import namedtuple
from aaa_org_helpers import *
from aaa_zsindy import *
from matplotlib import rcParams
rcParams['mathtext.fontset']='cm'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
Sol = namedtuple("Sol", ["t", "y"])

def convex_hull_from_df(df, a_col, k_col, step):
    pts = df[[a_col, k_col]].to_numpy(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        raise ValueError("Not enough points to form a convex hull.")
    hull = ConvexHull(pts)
    hull_poly = pts[hull.vertices]
    # grid over bbox
    (xmin, ymin) = hull_poly.min(axis=0)
    (xmax, ymax) = hull_poly.max(axis=0)
    A_grid = np.arange(xmin, xmax + step, step)
    K_grid = np.arange(ymin, ymax + step, step)
    AA, KK = np.meshgrid(A_grid, K_grid)
    # mask by polygon
    mask = Path(hull_poly).contains_points(
        np.c_[AA.ravel(), KK.ravel()]
    ).reshape(AA.shape)
    return hull_poly, A_grid, K_grid, AA, KK, mask

def grid_raw_field(df, a_col, k_col, ad_col, kd_col, AA, KK, mask):
    sub = df[[a_col, k_col, ad_col, kd_col]].copy()
    for c in [a_col, k_col, ad_col, kd_col]:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna()
    if sub.empty:
        return None, None, None

    P  = sub[[a_col, k_col]].to_numpy(float)
    Uv = sub[ad_col].to_numpy(float)
    Vv = sub[kd_col].to_numpy(float)

    U_lin = griddata(P, Uv, (AA, KK), method='linear')
    V_lin = griddata(P, Vv, (AA, KK), method='linear')
    U_near = griddata(P, Uv, (AA, KK), method='nearest')
    V_near = griddata(P, Vv, (AA, KK), method='nearest')

    U = np.where(np.isfinite(U_lin), U_lin, U_near).astype(float)
    V = np.where(np.isfinite(V_lin), V_lin, V_near).astype(float)

    U = np.where(mask, U, np.nan)
    V = np.where(mask, V, np.nan)
    speed = np.where(mask, np.hypot(U, V), np.nan)
    return U, V, speed

def vector_field_from_mu(mu, B, AA, KK):
    c = mu[0,:]
    U = c[0] + B[0,0]*AA + B[0,1]*KK
    V = c[1] + B[1,0]*AA + B[1,1]*KK
    return U, V

def _pairwise_vectors(df,
                      id_col="Patient_ID",
                      time_col="Years_mean",
                      a_col="SurfaceArea_Norm_mean",
                      k_col="IntGaussian_Fluct_Norm_mean"):
    """
    Build pairwise vectors for every consecutive scan per patient
    using normalized A, K and Years_mean.

    Returns dict:
      'A0','K0'        : starting points (scan i)
      'dA','dK'        : displacements to scan i+1 (for RAW arrows)
      'U','V'          : per-year rates (Δ/Δt) at start (for INTERPOLATED field)
    """
    need = [id_col, time_col, a_col, k_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in dataframe.")

    sub = df[need].copy()
    for c in [time_col, a_col, k_col]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return {k: None for k in ["A0","K0","dA","dK","U","V"]}

    sub = sub.sort_values([id_col, time_col], kind="mergesort")
    # shift within each patient to create pairs
    for col in (time_col, a_col, k_col):
        sub[f"{col}_next"] = sub.groupby(id_col, sort=False)[col].shift(-1)

    # keep only rows that have a "next" scan
    pairs = sub.dropna(
        subset=[f"{time_col}_next", f"{a_col}_next", f"{k_col}_next"]
    ).copy()
    if pairs.empty:
        return {k: None for k in ["A0","K0","dA","dK","U","V"]}

    dt = (pairs[f"{time_col}_next"] - pairs[time_col]).to_numpy(float)
    good = np.isfinite(dt) & (dt > 0)
    if not np.any(good):
        return {k: None for k in ["A0","K0","dA","dK","U","V"]}

    A0 = pairs[a_col].to_numpy(float)[good]
    K0 = pairs[k_col].to_numpy(float)[good]
    dA = (pairs[f"{a_col}_next"].to_numpy(float) - pairs[a_col].to_numpy(float))[good]
    dK = (pairs[f"{k_col}_next"].to_numpy(float) - pairs[k_col].to_numpy(float))[good]
    U  = dA / dt[good]
    V  = dK / dt[good]
    return {"A0":A0, "K0":K0, "dA":dA, "dK":dK, "U":U, "V":V}

def _grid_from_pairs(A0, K0, U, V, AA, KK, mask):
    """
    Interpolate a vector field from pairwise per-year rates (U,V),
    defined at starting locations (A0,K0).
    """
    if A0 is None:
        return None, None, None

    P = np.column_stack([A0, K0]).astype(float)
    Uv = np.asarray(U, float)
    Vv = np.asarray(V, float)

    U_lin = griddata(P, Uv, (AA, KK), method="linear")
    V_lin = griddata(P, Vv, (AA, KK), method="linear")
    U_near = griddata(P, Uv, (AA, KK), method="nearest")
    V_near = griddata(P, Vv, (AA, KK), method="nearest")

    Ug = np.where(np.isfinite(U_lin), U_lin, U_near).astype(float)
    Vg = np.where(np.isfinite(V_lin), V_lin, V_near).astype(float)

    Ug = np.where(mask, Ug, np.nan)
    Vg = np.where(mask, Vg, np.nan)
    Sg = np.where(mask, np.hypot(Ug, Vg), np.nan)
    return Ug, Vg, Sg

def add_eigendirections_equal_screen(ax, x0, B, pixels=None, frac=0.18, **kw):
    """
    Draw eigenvectors of B at x0 with a constant length relative to the axes size.
    """
    w, V = np.linalg.eig(B)
    V = np.real(V)
    V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    bb = ax.get_window_extent()
    L_pixels = (pixels if pixels is not None else frac * min(bb.width, bb.height))

    x0 = np.asarray(x0)
    p0 = ax.transData.transform(x0)

    base_style = {
        "color":"k","lw":2.2,"alpha":0.95,"solid_capstyle":"round"
    } | kw

    for j in range(2):
        v_screen = ax.transData.transform(x0 + V[:, j]) - p0
        n = float(np.hypot(v_screen[0], v_screen[1]))
        if n < 1e-12:
            continue
        u = v_screen / n

        pL = p0 - u * (L_pixels / 2.0)
        pR = p0 + u * (L_pixels / 2.0)

        dL = ax.transData.inverted().transform(pL)
        dR = ax.transData.inverted().transform(pR)
        ax.plot([dL[0], dR[0]], [dL[1], dR[1]], **base_style)

def _rbf_epsilon(P):
    """Median distance to the 3rd NN as a robust scale."""
    if len(P) < 4:
        return 1.0
    nn = NearestNeighbors(n_neighbors=min(4, len(P)), algorithm='auto').fit(P)
    dists, _ = nn.kneighbors(P)  # shape (n, k)
    return float(np.median(dists[:, -1]))  # 3rd NN

def _grid_from_pairs_rbf(A0, K0, U, V, AA, KK, mask,
                         kernel='thin_plate_spline',
                         epsilon=None,
                         smoothing=1e-2):
    """
    Smooth RBF interpolation of vector field defined at starts (A0,K0) with rates (U,V).
    Returns Ug, Vg, Sg on grid; NaNs outside mask.
    """
    if A0 is None:
        return None, None, None
    # Prepare data
    P = np.column_stack([A0, K0]).astype(float)
    Uv = np.asarray(U, float)
    Vv = np.asarray(V, float)
    # Remove non-finite rows
    good = np.isfinite(P).all(axis=1) & np.isfinite(Uv) & np.isfinite(Vv)
    P    = P[good]
    Uv   = Uv[good]
    Vv   = Vv[good]
    if P.shape[0] < 3:
        # too few points to build a reliable RBF surface
        Ug = np.full_like(AA, np.nan, dtype=float)
        Vg = np.full_like(KK, np.nan, dtype=float)
        Sg = np.full_like(AA, np.nan, dtype=float)
        return Ug, Vg, Sg
    # Scale coordinates (optional but helpful if A,K ranges differ a lot)
    # Simple z-score:
    mu = P.mean(axis=0); sd = P.std(axis=0) + 1e-12
    Pn = (P - mu) / sd
    # Grid points to evaluate
    G = np.column_stack([AA.ravel(), KK.ravel()])
    Gn = (G - mu) / sd
    # Choose epsilon if not given
    eps = _rbf_epsilon(Pn) if (epsilon is None) else float(epsilon)
    # Fit separate RBFs for U and V
    # Note: RBFInterpolator supports multi-output if you stack, but
    # doing two fits keeps it explicit.
    rbf_U = RBFInterpolator(Pn, Uv, kernel=kernel, epsilon=eps, smoothing=smoothing)
    rbf_V = RBFInterpolator(Pn, Vv, kernel=kernel, epsilon=eps, smoothing=smoothing)
    Ug = rbf_U(Gn).reshape(AA.shape)
    Vg = rbf_V(Gn).reshape(AA.shape)
    # Mask outside domain
    Ug = np.where(mask, Ug, np.nan)
    Vg = np.where(mask, Vg, np.nan)
    Sg = np.where(mask, np.hypot(Ug, Vg), np.nan)
    return Ug, Vg, Sg

def plot_stream_basic(
    df_regr, df_stable,
    mu_regr, mu_stab,
    B_regr, B_stab,
    a_col="SurfaceArea_Norm_mean",
    k_col="IntGaussian_Fluct_Norm_mean",
    id_col="Patient_ID",
    time_col="Years_mean",
    step_size=0.25,
    arrowsize=1.0,
    quiver_scale=1.0,
    quiver_width=0.003,
    raw_blue="#1f77b4",
    raw_red ="#d62728",
    A_max=6.0,
    K_max=10.0,
    interp_smoothing=1e-2,
    smoothing_kernel='thin_plate_spline',
    dpi=100,
):
    """
    2×3 figure:

      Row 1 (Regressing):  Raw (CT) | Interpolated (CT, RBF) | Model (pipeline_results)
      Row 2 (Stable):      Raw (CT) | Interpolated (CT, RBF) | Model (pipeline_results)

    Column 1 arrows:
      Tail at initial (A0, K0), head at final (A0 + dA, K0 + dK).
    """

    # --- filter to plotting window (normalized A, K) using CT data ---
    regr_f = df_regr[(df_regr[a_col] < A_max) & (df_regr[k_col] < K_max)].copy()
    stab_f = df_stable[(df_stable[a_col] < A_max) & (df_stable[k_col] < K_max)].copy()

    # grid & hull for each cohort (based on CT data)
    hull_poly_regr, A_grid_regr, K_grid_regr, AA_regr, KK_regr, mask_regr = \
        convex_hull_from_df(regr_f, a_col, k_col, step_size)
    hull_poly_stab, A_grid_stab, K_grid_stab, AA_stab, KK_stab, mask_stab = \
        convex_hull_from_df(stab_f, a_col, k_col, step_size)

    poly_regr = np.vstack([hull_poly_regr, hull_poly_regr[0]])
    poly_stab = np.vstack([hull_poly_stab, hull_poly_stab[0]])

    # ---- Build pairwise vectors from the CT datasets ----
    PW_regr = _pairwise_vectors(
        df_regr, id_col=id_col, time_col=time_col,
        a_col=a_col, k_col=k_col
    )
    PW_stab = _pairwise_vectors(
        df_stable, id_col=id_col, time_col=time_col,
        a_col=a_col, k_col=k_col
    )

    # Interpolated fields from CT pairwise rates (for columns 2) via RBF
    Uraw_regr, Vraw_regr, Sraw_regr = _grid_from_pairs_rbf(
        PW_regr["A0"], PW_regr["K0"], PW_regr["U"], PW_regr["V"],
        AA_regr, KK_regr, mask_regr,
        kernel=smoothing_kernel,
        smoothing=interp_smoothing
    )
    Uraw_stab, Vraw_stab, Sraw_stab = _grid_from_pairs_rbf(
        PW_stab["A0"], PW_stab["K0"], PW_stab["U"], PW_stab["V"],
        AA_stab, KK_stab, mask_stab,
        kernel=smoothing_kernel,
        smoothing=interp_smoothing
    )

    # MODEL fields (right column) from pipeline_results mu's
    Um_regr, Vm_regr = vector_field_from_mu(mu_regr, B_regr, AA_regr, KK_regr)
    Um_regr[~mask_regr] = np.nan
    Vm_regr[~mask_regr] = np.nan

    Um_stab, Vm_stab = vector_field_from_mu(mu_stab, B_stab, AA_stab, KK_stab)
    Um_stab[~mask_stab] = np.nan
    Vm_stab[~mask_stab] = np.nan

    Sm_regr = np.where(mask_regr, np.hypot(Um_regr, Vm_regr), np.nan)
    Sm_stab = np.where(mask_stab, np.hypot(Um_stab, Vm_stab), np.nan)

    # color scale across everything
    mags = [Sm_regr, Sm_stab]
    if Sraw_regr is not None:
        mags.append(Sraw_regr)
    if Sraw_stab is not None:
        mags.append(Sraw_stab)
    vmin = np.nanmin([np.nanmin(m) for m in mags])
    vmax = 1.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrBr

    # figure
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.5),
        sharex=True, sharey=True, dpi=dpi
    )
    fig.subplots_adjust(
        left=0.07, right=0.92, top=0.95, bottom=0.10,
        wspace=0.12, hspace=0.28
    )

    # ------- Row 1: Regressing -------

    # Col 1: RAW arrows (true displacement from scan i to i+1)
    ax = axes[0,0]
    ax.plot(poly_regr[:,0], poly_regr[:,1], color='k', lw=1.0, alpha=0.9)
    if PW_regr["A0"] is not None:
        ax.quiver(
            PW_regr["A0"], PW_regr["K0"],
            PW_regr["dA"], PW_regr["dK"],
            angles='xy', scale_units='xy', scale=1.0,   # tail at (A0,K0), head at (A0+dA,K0+dK)
            width=quiver_width, color=raw_blue, alpha=0.95
        )
    ax.text(0.5, 1.02, "Regressing Sac:Raw", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # Col 2: INTERPOLATED CT field from pairwise rates (RBF-smoothed)
    ax = axes[0,1]
    if Uraw_regr is not None:
        ax.contourf(
            A_grid_regr, K_grid_regr, Sraw_regr,
            levels=15, cmap=cmap, norm=norm, alpha=0.6
        )
        ax.streamplot(
            A_grid_regr, K_grid_regr, Uraw_regr, Vraw_regr,
            color=raw_blue, linewidth=1.0,
            arrowsize=arrowsize, minlength=0.05, density=0.75
        )
    else:
        ax.text(0.02, 0.95, "No interpolated raw field",
                transform=ax.transAxes, ha='left', va='top', fontsize=12)
    ax.text(0.5, 1.02, "Regressing Sac:Interpolated", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # Col 3: MODEL field (from pipeline_results mu/B)
    ax = axes[0,2]
    hmap = ax.contourf(
        AA_regr, KK_regr, Sm_regr,
        levels=20, cmap=cmap, norm=norm, alpha=0.6
    )
    ax.streamplot(
        A_grid_regr, K_grid_regr, Um_regr, Vm_regr,
        color=raw_blue, linewidth=1.0,
        arrowsize=arrowsize, minlength=0.05, density=0.75
    )
    fp = get_fixed_points(mu_regr)
    add_eigendirections_equal_screen(ax, fp, B_regr, frac=0.35)
    ax.plot(fp[0], fp[1], marker='*', ms=18, color='white',
            mec='black', mew=1.6, zorder=6)
    heatmap_handle = hmap
    ax.text(0.5, 1.02, "Regressing Sac:Model", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # ------- Row 2: Stable -------

    # Col 1: RAW arrows (true displacement from scan i to i+1)
    ax = axes[1,0]
    ax.plot(poly_stab[:,0], poly_stab[:,1], color='k', lw=1.0, alpha=0.9)
    if PW_stab["A0"] is not None:
        ax.quiver(
            PW_stab["A0"], PW_stab["K0"],
            PW_stab["dA"], PW_stab["dK"],
            angles='xy', scale_units='xy', scale=1.0,
            width=quiver_width, color=raw_red, alpha=0.95
        )
    ax.text(0.5, 1.02, "Stable Sac:Raw", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # Col 2: INTERPOLATED CT field (RBF-smoothed)
    ax = axes[1,1]
    if Uraw_stab is not None:
        ax.contourf(
            A_grid_stab, K_grid_stab, Sraw_stab,
            levels=15, cmap=cmap, norm=norm, alpha=0.6
        )
        ax.streamplot(
            A_grid_stab, K_grid_stab, Uraw_stab, Vraw_stab,
            color=raw_red, linewidth=1.0,
            arrowsize=arrowsize, minlength=0.05, density=0.75
        )
    else:
        ax.text(0.02, 0.95, "No interpolated raw field",
                transform=ax.transAxes, ha='left', va='top', fontsize=12)
    ax.text(0.5, 1.02, "Stable Sac:Interpolated", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # Col 3: MODEL (from pipeline_results)
    ax = axes[1,2]
    ax.contourf(
        AA_stab, KK_stab, Sm_stab,
        levels=20, cmap=cmap, norm=norm, alpha=0.6
    )
    ax.streamplot(
        A_grid_stab, K_grid_stab, Um_stab, Vm_stab,
        color=raw_red, linewidth=1.0,
        arrowsize=arrowsize, minlength=0.05, density=0.75
    )
    fp2 = get_fixed_points(mu_stab)
    add_eigendirections_equal_screen(ax, fp2, B_stab, frac=0.2)
    ax.plot(fp2[0], fp2[1], marker='*', ms=18, color='white',
            mec='black', mew=1.6, zorder=6)
    ax.text(0.5, 1.02, "Stable Sac:Model", color='k',
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontname="Times New Roman")

    # ---- axes formatting ----
    for i in range(2):
        for j in range(3):
            axes[i,j].set_xlim(2, 6)
            axes[i,j].set_ylim(-0.5, 10)
            axes[i,j].set_xticks(np.arange(1, 7, 1))
            axes[i,j].set_yticks(np.arange(0, 11, 2))
            axes[i,j].grid(True, linestyle="--", alpha=0.6)
            axes[i,j].tick_params(labelsize=14)

    for j in range(3):
        axes[1,j].set_xlabel(r"$\widetilde{A}$", fontsize=18)
    for i in range(2):
        axes[i,0].set_ylabel(r"$\widetilde{\delta K}$", fontsize=18)

    for i in range(2):
        for j in range(3):
            for label in axes[i,j].get_xticklabels() + axes[i,j].get_yticklabels():
                label.set_fontname("Times New Roman")

    # Legends
    handles_raw = [
        Line2D([0],[0], color='k', lw=2, label="Patient\nTrajectory"),
    ]
    axes[0,0].legend(
        handles=handles_raw, fontsize=9, loc="upper left",
        frameon=False, prop={'family': 'Times New Roman'}
    )

    handles_model = [
        Line2D([0],[0], color='black', lw=2.2, label="Eigenvectors"),
        Line2D([0],[0], marker='*', color='white', mec='black', mew=1.3, ms=12,
               linestyle='None', label="Fixed\nPoint")
    ]
    axes[0,2].legend(
        handles=handles_model, fontsize=9, loc="upper left",
        frameon=False, prop={'family': 'Times New Roman'}
    )

    # colorbar
    pos_mid_top  = axes[0,1].get_position()
    pos_mid_bot  = axes[1,1].get_position()
    pos_right_top= axes[0,2].get_position()
    pos_right_bot= axes[1,2].get_position()
    y0 = min(pos_mid_bot.y0, pos_right_bot.y0)
    y1 = max(pos_mid_top.y1, pos_right_top.y1)
    h  = y1 - y0
    cax = fig.add_axes([0.95, y0, 0.02, h])
    cb = fig.colorbar(heatmap_handle, cax=cax)
    cb.set_label("Vector magnitude", fontsize=16, fontname="Times New Roman")
    cb.ax.tick_params(labelsize=12, labelrotation=0)
    for label in cb.ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    plt.show()

def _rbf_fit_error_at_points(A0, K0, U, V,
                             kernel='thin_plate_spline',
                             smoothing=1e-2,
                             epsilon=None):
    """
    Fit RBFInterpolator for (U,V) on (A0,K0), then evaluate back at the same
    (A0,K0) and compute normalized RMSE between (U,V) and (U_hat,V_hat).
    """
    if A0 is None:
        return np.nan

    # stack coordinates
    P  = np.column_stack([A0, K0]).astype(float)
    Uv = np.asarray(U, float)
    Vv = np.asarray(V, float)

    # keep only finite rows
    good = np.isfinite(P).all(axis=1) & np.isfinite(Uv) & np.isfinite(Vv)
    P  = P[good]
    Uv = Uv[good]
    Vv = Vv[good]

    if P.shape[0] < 3:
        return np.nan

    # simple z-score scaling
    mu = P.mean(axis=0)
    sd = P.std(axis=0) + 1e-12
    Pn = (P - mu) / sd

    # pick epsilon if not provided
    eps = _rbf_epsilon(Pn) if (epsilon is None) else float(epsilon)

    # fit RBFs
    rbf_U = RBFInterpolator(Pn, Uv, kernel=kernel, epsilon=eps, smoothing=smoothing)
    rbf_V = RBFInterpolator(Pn, Vv, kernel=kernel, epsilon=eps, smoothing=smoothing)

    U_hat = rbf_U(Pn)
    V_hat = rbf_V(Pn)

    # vector RMSE
    resid_sq = (U_hat - Uv)**2 + (V_hat - Vv)**2
    rmse = np.sqrt(np.mean(resid_sq))

    data_mag_sq = Uv**2 + Vv**2
    data_rms = np.sqrt(np.mean(data_mag_sq)) + 1e-12

    nrmse = rmse / data_rms  # dimensionless
    return nrmse


def justify_interp_smoothing_rbf(
    df_regr,
    df_stable,
    *,
    a_col="SurfaceArea_Norm_mean",
    k_col="IntGaussian_Fluct_Norm_mean",
    id_col="Patient_ID",
    time_col="Years_mean",
    smoothing_values=None,
    kernel='thin_plate_spline',
    chosen_smoothing=None,
    raw_blue="#1f77b4",
    raw_red ="#d62728",
    dpi=100,
):

    if smoothing_values is None:
        smoothing_values = np.logspace(-4, 2, 25)

    smoothing_values = np.asarray(smoothing_values, float)

    # pairwise vectors from your CT data (same as in plot_stream_basic)
    PW_regr = _pairwise_vectors(
        df_regr, id_col=id_col, time_col=time_col,
        a_col=a_col, k_col=k_col
    )
    PW_stab = _pairwise_vectors(
        df_stable, id_col=id_col, time_col=time_col,
        a_col=a_col, k_col=k_col
    )

    nrmse_regr  = []
    nrmse_stab  = []

    for s in smoothing_values:
        err_r = _rbf_fit_error_at_points(
            PW_regr["A0"], PW_regr["K0"], PW_regr["U"], PW_regr["V"],
            kernel=kernel, smoothing=s
        )
        err_s = _rbf_fit_error_at_points(
            PW_stab["A0"], PW_stab["K0"], PW_stab["U"], PW_stab["V"],
            kernel=kernel, smoothing=s
        )
        nrmse_regr.append(err_r)
        nrmse_stab.append(err_s)

    nrmse_regr  = np.array(nrmse_regr, float)
    nrmse_stab  = np.array(nrmse_stab, float)

    # ---- make figure ----
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    ax.semilogx(
        smoothing_values, nrmse_regr,
        marker='o', color=raw_blue, label='Regressing sac'
    )
    ax.semilogx(
        smoothing_values, nrmse_stab,
        marker='s', color=raw_red, label='Stable sac'
    )

    if chosen_smoothing is not None:
        ax.axvline(
            chosen_smoothing, color='k', ls='--', lw=1.2,
        )

    ax.set_xlabel("RBF smoothing parameter", fontsize=12, fontname="Times New Roman")
    ax.set_ylabel("Variance Explained", fontsize=12, fontname="Times New Roman")

    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.tick_params(labelsize=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Times New Roman")

    ax.legend(frameon=False, fontsize=4, prop={"family": "Times New Roman"}, loc="upper left")

    plt.tight_layout()

    results = {
        "smoothing": smoothing_values,
        "nrmse_regr": nrmse_regr,
        "nrmse_stable": nrmse_stab,
    }

    return fig, ax, results

def DiffEq_2order(t, y, coef):
    """
    2nd-order polynomial dynamics in (A,K).

    Assumes the library monomial order is the standard prefix:
        [1, A, K, A^2, A K, K^2, ...]
    but *truncates* to the actual number of rows in `coef`.

    coef: array-like (n_terms, n_targets>=2) -> uses first 2 columns as [dA/dt, dK/dt].
    """
    A, K = y

    # Full degree-2 basis for 2 variables
    full_phi = np.array(
        [1.0,
         A,
         K,
         A**2,
         A * K,
         K**2],
        dtype=float
    )

    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef[:, None]

    n_terms, n_targets = coef.shape

    # Use only the first two targets as (A,K); ignore any extras
    if n_targets < 2:
        raise ValueError(f"DiffEq_2order: need at least 2 targets (A,K), got {n_targets}")
    if n_targets > 2:
        coef = coef[:, :2]    # keep first two columns
        n_targets = 2

    # Sanity check basis length vs terms
    if n_terms > full_phi.size:
        raise ValueError(
            f"DiffEq_2order: coef has {n_terms} terms, "
            f"but degree-2 basis only has {full_phi.size}"
        )

    # Truncate basis to the number of terms actually used
    phi = full_phi[:n_terms]

    dA_dt = float(phi @ coef[:, 0])
    dK_dt = float(phi @ coef[:, 1])
    return np.array([dA_dt, dK_dt], dtype=float)


def DiffEq_3order(t, y, coef):
    """
    3rd-order polynomial dynamics in (A,K).

    Assumes standard monomial prefix order:
        [1, A, K,
         A^2, A K, K^2,
         A^3, A^2 K, A K^2, K^3, ...]
    and truncates to len(coef).

    coef: array-like (n_terms, n_targets>=2) -> uses first 2 columns as [dA/dt, dK/dt].
    """
    A, K = y

    full_phi = np.array(
        [1.0,
         A,
         K,
         A**2,
         A * K,
         K**2,
         A**3,
         A**2 * K,
         A * K**2,
         K**3],
        dtype=float
    )

    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef[:, None]

    n_terms, n_targets = coef.shape

    # Use only the first two targets as (A,K); ignore any extras
    if n_targets < 2:
        raise ValueError(f"DiffEq_3order: need at least 2 targets (A,K), got {n_targets}")
    if n_targets > 2:
        coef = coef[:, :2]
        n_targets = 2

    if n_terms > full_phi.size:
        raise ValueError(
            f"DiffEq_3order: coef has {n_terms} terms, "
            f"but degree-3 basis only has {full_phi.size}"
        )

    phi = full_phi[:n_terms]

    dA_dt = float(phi @ coef[:, 0])
    dK_dt = float(phi @ coef[:, 1])
    return np.array([dA_dt, dK_dt], dtype=float)


def _zsindy_pipeline_core(
    regr_data,
    stable_data,
    poly_degree,
    max_terms,
    diff_eq_func=None,
    x0=None,
    x0_std=None,
    ens_trials=1000,
    colors=None,
    plot_bool=True,
):
    """
    Generalized SINDy pipeline used by zsindy_pipeline_2order and
    zsindy_pipeline_3order.

    For poly_degree >= 2, the deterministic trajectory `sol` is now
    generated with model.simulate(), so it uses exactly the same library
    and coefficient ordering as SINDy itself.
    """

    assert len(regr_data) == len(stable_data), "Lists must match length"
    N = len(regr_data)
    results_regr   = []
    results_stable = []

    for ds_list, res_list in [(regr_data, results_regr),
                              (stable_data, results_stable)]:
        for entry in ds_list:
            df    = entry['df'].sort_values('Years_mean')
            name  = entry['name']
            lam   = entry['lmbda']

            x = df[['SurfaceArea_Norm_mean',
                    'IntGaussian_Fluct_Norm_mean']].values
            t = df['Years_mean'].values
            xdot = df[['dSurfaceArea_Norm_mean/dYears_mean',
                       'dIntGaussian_Fluct_Norm_mean/dYears_mean']].values

            x0_use = x0 if x0 is not None else x[0]

            # noise/length-scale estimation (still linear [1, A, K] model)
            mu, sigma, rho = estimate_rho(x, xdot)
            A_fp, K_fp     = get_fixed_points(mu)

            # fit Z-SINDy
            model = ZSindy(
                poly_degree=poly_degree,
                lmbda=lam,
                max_num_terms=max_terms,
                rho=rho,
                variable_names=['A', 'K'],
            )
            model.fit(x, t, xdot)
            model.print()

            coef = np.asarray(model.coefficients())           # (2, n_terms)
            var  = np.asarray(model.coefficients_variance())  # same shape

            # -------- deterministic trajectory (sol) ----------
            # Use the SAME integrator as the ensemble: model.simulate.
            # This avoids any mismatch with internal feature ordering.
            t_sol = np.linspace(0, 10, 500)
            y_sol = model.simulate(x0_use, t_sol, coef)   # (len(t_sol), 2)
            sol   = Sol(t=t_sol, y=y_sol.T)               # mimic solve_ivp API

            # -------- ensemble sampling of coefficients + x0 ----------
            nd, nf = coef.shape  # (n_dims=2, n_features)
            if x0_std is None:
                x0_samps = np.tile(x0_use, (ens_trials, 1))
            else:
                x0_samps = np.random.normal(
                    x0_use, x0_std, size=(ens_trials, len(x0_use))
                )

            sd = np.sqrt(var)
            c_samps = np.random.normal(
                coef.reshape(-1),
                sd.reshape(-1),
                size=(ens_trials, nd * nf)
            ).reshape(ens_trials, nd, nf)

            ens = np.stack([
                model.simulate(x0_samps[k], t, c_samps[k])
                for k in range(ens_trials)
            ])

            res_list.append({
                'name':  name,
                'df':    df,
                't':     t,
                'x':     x,
                'xdot':  xdot,
                'sol':   sol,
                'coef':  coef,
                'sd':    sd,
                'A_fp':  A_fp,
                'K_fp':  K_fp,
                'model': model,
                'x0':    x0_use,
                'rho':   rho,
                'sigma': sigma,
                'mu':    mu,
                'B':     mu[1:, :].T,   # still the linear surrogate if you want it
                'ens':   ens,
            })

    names_regr   = [r['name'] for r in results_regr]
    names_stable = [s['name'] for s in results_stable]

    # ---------------- plotting (same structure as your original) ----------------
    if plot_bool:
        # --- Figure 1: trajectories + fixed points ---
        fig1, axes1 = plt.subplots(
            N, 2, figsize=(12, 4 * N),
            facecolor='whitesmoke', squeeze=False
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        for i in range(N):
            axA, axK = axes1[i, 0], axes1[i, 1]
            axA.text(
                0.02, 0.95,
                f"{names_regr[i]}",
                transform=axA.transAxes,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7)
            )

            for results, color, tag in [
                (results_regr[i],   colors['Regr'],   'regr'),
                (results_stable[i], colors['Stable'], 'stable'),
            ]:
                name   = results['name']
                sol    = results['sol']
                x      = results['x']
                A_fp   = results['A_fp']
                K_fp   = results['K_fp']

                # --- Size (A) plot ---
                axA.plot(sol.t, sol.y[0], '-', lw=3, color=color,
                         label=f"{name} ({tag})")
                axA.scatter(results['t'], x[:, 0], color=color, s=20)
                axA.axhline(A_fp, color=color, ls='--', lw=2)
                axA.set_title("Size")
                axA.set_ylabel('A')
                axA.set_ylim(0, 10)

                # --- Shape (δK) plot ---
                axK.plot(sol.t, sol.y[1], '-', lw=3, color=color)
                axK.scatter(results['t'], x[:, 1], color=color, s=20)
                axK.axhline(K_fp, color=color, ls='--', lw=2)
                axK.set_title("Shape")
                axK.set_ylabel('K')
                axK.set_ylim(0, 15)

            if i == 0:
                axA.legend(loc='upper right')
            axA.set_xlabel('Years')
            axK.set_xlabel('Years')

        plt.tight_layout()
        plt.show()

        # --- Figure 2: ensemble trajectories + marginals ---
        fig2 = plt.figure(figsize=(18, 4 * N), facecolor='whitesmoke')
        outer_gs = gridspec.GridSpec(N, 4, figure=fig2, hspace=0.5, wspace=0.3)

        for i in range(N):
            res_r = results_regr[i]
            res_s = results_stable[i]

            # --- ensemble for Regressing ---
            sol_r, coef_r, sd_r = res_r['sol'], res_r['coef'], res_r['sd']
            model_r, x0_r       = res_r['model'], res_r['x0']
            if x0_std is None:
                x0_samps_r = np.tile(x0_r, (ens_trials, 1))
            else:
                x0_samps_r = np.random.normal(
                    x0_r, x0_std, size=(ens_trials, len(x0_r))
                )
            nd, nf = coef_r.shape
            c_samps_r = np.random.normal(
                coef_r.reshape(-1),
                sd_r.reshape(-1),
                size=(ens_trials, nd * nf)
            ).reshape(ens_trials, nd, nf)
            ens_r = np.stack([
                model_r.simulate(x0_samps_r[k], sol_r.t, c_samps_r[k])
                for k in range(ens_trials)
            ])

            # --- ensemble for Stable ---
            sol_s, coef_s, sd_s = res_s['sol'], res_s['coef'], res_s['sd']
            model_s, x0_s0      = res_s['model'], res_s['x0']
            if x0_std is None:
                x0_samps_s = np.tile(x0_s0, (ens_trials, 1))
            else:
                x0_samps_s = np.random.normal(
                    x0_s0, x0_std, size=(ens_trials, len(x0_s0))
                )
            nd, nf = coef_s.shape
            c_samps_s = np.random.normal(
                coef_s.reshape(-1),
                sd_s.reshape(-1),
                size=(ens_trials, nd * nf)
            ).reshape(ens_trials, nd, nf)
            ens_s = np.stack([
                model_s.simulate(x0_samps_s[k], sol_s.t, c_samps_s[k])
                for k in range(ens_trials)
            ])

            # --- Panel 1: A ensemble ---
            axA = fig2.add_subplot(outer_gs[i, 0])
            axA.plot(sol_r.t, sol_r.y[0], '--', color='k', lw=1)
            axA.plot(sol_s.t, sol_s.y[0], '--', color='k', lw=1)
            axA.plot(sol_r.t, ens_r[:, :, 0].T,
                     color=colors['Regr'], alpha=0.01)
            axA.plot(sol_s.t, ens_s[:, :, 0].T,
                     color=colors['Stable'], alpha=0.01)
            axA.set_title("A SINDy ensemble")
            axA.set_ylabel('A')
            axA.scatter([], [], color=colors['Regr'],
                        label='Regressed', marker='o', s=50)
            axA.scatter([], [], color=colors['Stable'],
                        label='Stable',   marker='o', s=50)
            axA.legend(loc='upper right')
            axA.set_xlabel('Years')
            axA.set_ylim(0, 10)

            # --- Panel 2: K ensemble ---
            axK = fig2.add_subplot(outer_gs[i, 1])
            axK.plot(sol_r.t, sol_r.y[1], '--', color='k', lw=1)
            axK.plot(sol_s.t, sol_s.y[1], '--', color='k', lw=1)
            axK.plot(sol_r.t, ens_r[:, :, 1].T,
                     color=colors['Regr'], alpha=0.01)
            axK.plot(sol_s.t, ens_s[:, :, 1].T,
                     color=colors['Stable'], alpha=0.01)
            axK.set_title("δK SINDy ensemble")
            axK.set_ylabel('K')
            axK.scatter([], [], color=colors['Regr'],
                        label='Regressed', marker='o', s=50)
            axK.scatter([], [], color=colors['Stable'],
                        label='Stable',   marker='o', s=50)
            axK.legend(loc='upper right')
            axK.set_xlabel('Years')
            axK.set_ylim(0, 15)

            # --- Panel 3: A distributions at t=0,2.5,5 ---
            innerA = outer_gs[i, 2].subgridspec(3, 1, hspace=0.3)
            for k_idx, tp in enumerate([0, 2.5, 5]):
                ax = fig2.add_subplot(innerA[k_idx])
                idx = np.argmin(np.abs(sol_r.t - tp))
                sns.distplot(
                    ens_r[:, idx, 0],
                    hist=False, kde=True,
                    color=colors['Regr'], ax=ax
                )
                sns.distplot(
                    ens_s[:, idx, 0],
                    hist=False, kde=True,
                    color=colors['Stable'], ax=ax
                )
                ax.set_xlim(0, 10)
                ax.set_ylabel(f"t={tp}")
                ax.set_yticks([]); ax.set_yticklabels([])
                if k_idx < 2:
                    ax.set_xticks([]); ax.set_xticklabels([])
                if k_idx == 0:
                    ax.set_title("Size (A) distributions")

            # --- Panel 4: K distributions at t=0,2.5,5 ---
            innerK = outer_gs[i, 3].subgridspec(3, 1, hspace=0.3)
            for k_idx, tp in enumerate([0, 2.5, 5]):
                ax = fig2.add_subplot(innerK[k_idx])
                idx = np.argmin(np.abs(sol_r.t - tp))
                sns.distplot(
                    ens_r[:, idx, 1],
                    hist=False, kde=True,
                    color=colors['Regr'], ax=ax
                )
                sns.distplot(
                    ens_s[:, idx, 1],
                    hist=False, kde=True,
                    color=colors['Stable'], ax=ax
                )
                ax.set_xlim(0, 10)
                ax.set_ylabel(f"t={tp}")
                ax.set_yticks([]); ax.set_yticklabels([])
                if k_idx < 2:
                    ax.set_xticks([]); ax.set_xticklabels([])
                if k_idx == 0:
                    ax.set_title("Shape (δK) distributions")

        plt.tight_layout()
        plt.show()

    regr_rows = []
    for res in results_regr:
        t = res['t']
        ens = res['ens']
        for trial in range(ens.shape[0]):
            for j, time in enumerate(t):
                regr_rows.append({
                    'dataset': res['name'],
                    'trial': trial,
                    'Years_mean': time,
                    'SurfaceArea_Norm_mean': ens[trial, j, 0],
                    'IntGaussian_Fluct_Norm_mean': ens[trial, j, 1],
                })
    regr_ens = pd.DataFrame(regr_rows)

    stable_rows = []
    for res in results_stable:
        t = res['t']
        ens = res['ens']
        for trial in range(ens.shape[0]):
            for j, time in enumerate(t):
                stable_rows.append({
                    'dataset': res['name'],
                    'trial': trial,
                    'Years_mean': time,
                    'SurfaceArea_Norm_mean': ens[trial, j, 0],
                    'IntGaussian_Fluct_Norm_mean': ens[trial, j, 1],
                })
    stable_ens = pd.DataFrame(stable_rows)

    all_results = results_regr + results_stable
    return all_results, regr_ens, stable_ens

def zsindy_pipeline_2order(
    regr_data,
    stable_data,
    max_terms=None,
    x0=None,
    x0_std=None,
    ens_trials=1000,
    colors=None,
    plot_bool=True,
):
    """
    Convenience wrapper for a 2nd-order polynomial SINDy model
    in (A, K) with up to 6 monomials:
        [1, A, K, A^2, A K, K^2]
    By default, uses max_terms = 6 (full library).
    """
    if max_terms is None:
        max_terms = 6  # full 2nd-order library for 2 variables

    return _zsindy_pipeline_core(
        regr_data=regr_data,
        stable_data=stable_data,
        poly_degree=2,
        max_terms=max_terms,
        diff_eq_func=DiffEq_2order,
        x0=x0,
        x0_std=x0_std,
        ens_trials=ens_trials,
        colors=colors,
        plot_bool=plot_bool,
    )


def zsindy_pipeline_3order(
    regr_data,
    stable_data,
    max_terms=None,
    x0=None,
    x0_std=None,
    ens_trials=1000,
    colors=None,
    plot_bool=True,
):
    """
    Convenience wrapper for a 3rd-order polynomial SINDy model
    in (A, K) with up to 10 monomials:
        [1, A, K, A^2, A K, K^2, A^3, A^2 K, A K^2, K^3]
    By default, uses max_terms = 10 (full library).
    """
    if max_terms is None:
        max_terms = 10  # full 3rd-order library for 2 variables

    return _zsindy_pipeline_core(
        regr_data=regr_data,
        stable_data=stable_data,
        poly_degree=3,
        max_terms=max_terms,
        diff_eq_func=DiffEq_3order,
        x0=x0,
        x0_std=x0_std,
        ens_trials=ens_trials,
        colors=colors,
        plot_bool=plot_bool,
    )

def _theta_from_zsindy(z, x):
    """
    Return the library/design matrix Theta for a ZSindy model.

    - If z.Theta is already computed and matches N, reuse it.
    - Otherwise, call z.get_library_matrix(x).
    """
    x = np.asarray(x)
    N = x.shape[0]

    # Prefer the cached Theta from the last fit if the size matches
    if hasattr(z, "Theta") and z.Theta is not None:
        Theta = np.asarray(z.Theta)
        if Theta.shape[0] == N:
            return Theta

    # Otherwise, build fresh using the ZSindy API
    if hasattr(z, "get_library_matrix") and callable(z.get_library_matrix):
        return np.asarray(z.get_library_matrix(x))

    raise AttributeError(
        "ZSindy model has neither a matching `Theta` nor a callable `get_library_matrix(x)`."
    )

def _predict_with_coeffs(z, x, t, coeffs_terms_by_targets):
    """
    Generic prediction with explicit coefficients using ZSindy.

    - coeffs_terms_by_targets is (n_features, n_dims) (as returned by _get_coeffs_and_vars).
    - Returns yhat with shape (N, n_dims).
    """
    Theta = _theta_from_zsindy(z, x)      # (N, n_features)
    return Theta @ coeffs_terms_by_targets  # (N, n_dims)

def vector_field_from_sindy(z, AA, KK):
    """
    Build a vector field (U,V) on a grid (AA, KK) from a fitted ZSindy model.

    Parameters
    ----------
    z   : ZSindy instance (already fit).
    AA  : 2D array of A-coordinates (same shape as KK).
    KK  : 2D array of K-coordinates.

    Returns
    -------
    U, V : 2D arrays (same shape as AA/KK) giving dA/dt and dK/dt.
    """
    # Flatten grid into N x 2 state array
    A_flat = AA.ravel()
    K_flat = KK.ravel()
    X_flat = np.column_stack([A_flat, K_flat])  # (N, 2)

    # Library matrix for these points
    Theta = z.get_library_matrix(X_flat)        # (N, n_features)

    # Coefficients: shape (n_dims, n_features)
    C = np.asarray(z.coefficients())           # (2, n_features) here

    # Predict derivatives: yhat has shape (N, 2)
    Y_flat = Theta @ C.T                       # (N, 2)

    U = Y_flat[:, 0].reshape(AA.shape)
    V = Y_flat[:, 1].reshape(AA.shape)
    return U, V

def _sindy_rhs(z, x):
    """
    Evaluate the ZSindy ODE right-hand side at state x.

    x : array-like of shape (2,) or (n_dims,)
    returns: 1D array of shape (n_dims,)
    """
    x = np.asarray(x).reshape(1, -1)        # (1, n_dims)
    Theta = z.get_library_matrix(x)         # (1, n_features)
    C = np.asarray(z.coefficients())        # (n_dims, n_features)
    y = Theta @ C.T                         # (1, n_dims)
    return y.ravel()

def jacobian_from_sindy_fd(z, x0, eps=1e-5):
    """
    Finite-difference Jacobian of the ZSindy vector field at x0.

    Parameters
    ----------
    z   : ZSindy instance (already fit).
    x0  : iterable of length n_dims (e.g., [A*, K*] for 2D).
    eps : float, finite difference step.

    Returns
    -------
    J : (n_dims, n_dims) array, where J[i,j] = d f_i / d x_j at x0.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    f0 = _sindy_rhs(z, x0)
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        dx = np.zeros_like(x0)
        dx[j] = eps
        f_plus  = _sindy_rhs(z, x0 + dx)
        f_minus = _sindy_rhs(z, x0 - dx)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return J

def plot_stream_advanced(
    df_regr,
    df_stable,
    model_regr,
    model_stab,
    a_col="SurfaceArea_Norm_mean",
    k_col="IntGaussian_Fluct_Norm_mean",
    step_size=0.05,
    arrowsize=1.0,
    raw_blue="#1f77b4",
    raw_red ="#d62728",
    A_max=6.0,
    K_max=10.0,
    dpi=100,
):
    """
    2×1 figure:

      Row 1 (Regressing):  Nonlinear SINDy model vector field
      Row 2 (Stable):      Nonlinear SINDy model vector field

    - No raw arrows, no interpolated RBF field.
    - No fixed points or eigenvectors.
    """

    # --- restrict CT data to plotting window ---
    regr_f = df_regr[(df_regr[a_col] < A_max) & (df_regr[k_col] < K_max)].copy()
    stab_f = df_stable[(df_stable[a_col] < A_max) & (df_stable[k_col] < K_max)].copy()

    # --- convex hull + grid for each cohort (based on CT data) ---
    hull_poly_regr, A_grid_regr, K_grid_regr, AA_regr, KK_regr, mask_regr = \
        convex_hull_from_df(regr_f, a_col, k_col, step_size)
    hull_poly_stab, A_grid_stab, K_grid_stab, AA_stab, KK_stab, mask_stab = \
        convex_hull_from_df(stab_f, a_col, k_col, step_size)

    poly_regr = np.vstack([hull_poly_regr, hull_poly_regr[0]])
    poly_stab = np.vstack([hull_poly_stab, hull_poly_stab[0]])

    # --- Nonlinear SINDy model fields ---
    Um_regr, Vm_regr = vector_field_from_sindy(model_regr, AA_regr, KK_regr)
    Um_stab, Vm_stab = vector_field_from_sindy(model_stab, AA_stab, KK_stab)

    # mask outside convex hull
    Um_regr = np.where(mask_regr, Um_regr, np.nan)
    Vm_regr = np.where(mask_regr, Vm_regr, np.nan)
    Um_stab = np.where(mask_stab, Um_stab, np.nan)
    Vm_stab = np.where(mask_stab, Vm_stab, np.nan)

    Sm_regr = np.where(mask_regr, np.hypot(Um_regr, Vm_regr), np.nan)
    Sm_stab = np.where(mask_stab, np.hypot(Um_stab, Vm_stab), np.nan)

    # --- shared color normalization across both magnitude maps ---
    mags = [Sm_regr, Sm_stab]
    vmin = np.nanmin([np.nanmin(m) for m in mags])
    vmax = 1.5   # keep your hard cap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrBr

    # --- figure setup: 2 rows, 1 column ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(6.5, 6.5),
        sharex=True, sharey=True,
        dpi=dpi
    )
    fig.subplots_adjust(
        left=0.10, right=0.88, top=0.95, bottom=0.10,
        hspace=0.28
    )

    # Ensure axes is indexable as [0], [1]
    axes = np.atleast_1d(axes)

    # ==========================================================
    # Row 1: Regressing SINDy model field
    # ==========================================================
    ax = axes[0]
    hmap = ax.contourf(
        AA_regr, KK_regr, Sm_regr,
        levels=20, cmap=cmap, norm=norm, alpha=0.6
    )
    ax.streamplot(
        A_grid_regr, K_grid_regr, Um_regr, Vm_regr,
        color=raw_blue, linewidth=1.0,
        arrowsize=arrowsize, minlength=0.05, density=0.75
    )
    ax.plot(poly_regr[:, 0], poly_regr[:, 1], color='k', lw=1.0, alpha=0.9)
    ax.text(
        0.5, 1.02,
        "Regressing Sac:Model",
        color='k',
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=16,
        fontname="Times New Roman"
    )

    # ==========================================================
    # Row 2: Stable SINDy model field
    # ==========================================================
    ax = axes[1]
    ax.contourf(
        AA_stab, KK_stab, Sm_stab,
        levels=20, cmap=cmap, norm=norm, alpha=0.6
    )
    ax.streamplot(
        A_grid_stab, K_grid_stab, Um_stab, Vm_stab,
        color=raw_red, linewidth=1.0,
        arrowsize=arrowsize, minlength=0.05, density=0.75
    )
    ax.plot(poly_stab[:, 0], poly_stab[:, 1], color='k', lw=1.0, alpha=0.9)
    ax.text(
        0.5, 1.02,
        "Stable Sac:Model",
        color='k',
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=16,
        fontname="Times New Roman"
    )

    # ==========================================================
    # Axes formatting
    # ==========================================================
    for ax in axes:
        ax.set_xlim(2, 6)
        ax.set_ylim(-0.5, 10)
        ax.set_xticks(np.arange(1, 7, 1))
        ax.set_yticks(np.arange(0, 11, 2))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")

    # Labels: x on bottom axis, y on both
    axes[1].set_xlabel(r"$\widetilde{A}$", fontsize=18, fontname="Times New Roman")
    axes[0].set_ylabel(r"$\widetilde{\delta K}$", fontsize=18, fontname="Times New Roman")
    axes[1].set_ylabel(r"$\widetilde{\delta K}$", fontsize=18, fontname="Times New Roman")

    # ==========================================================
    # Colorbar
    # ==========================================================
    # Place a vertical colorbar to the right of the single column
    pos_top = axes[0].get_position()
    pos_bot = axes[1].get_position()
    y0 = pos_bot.y0
    y1 = pos_top.y1
    h  = y1 - y0
    cax = fig.add_axes([0.90, y0, 0.02, h])
    cb = fig.colorbar(hmap, cax=cax)
    cb.set_label("Vector magnitude", fontsize=16, fontname="Times New Roman")
    cb.ax.tick_params(labelsize=12, labelrotation=0)
    for label in cb.ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    plt.show()
    return fig, axes

def affine_from_mu(mu):
    """
    mu: array-like, shape (3, 2)
        rows: [1, A, K]
        cols: [A_dot, K_dot]
    Returns
    -------
    b : (2,) constant drift
    J : (2,2) Jacobian / linear operator
    """
    mu = np.asarray(mu, dtype=float)
    b = mu[0, :]  # constants for [A_dot, K_dot]

    # J maps [A, K] -> [A_dot, K_dot]
    # row i = equation for xdot_i, col j = derivative wrt state_j
    J = np.array([
        [mu[1, 0], mu[2, 0]],  # d(A_dot)/dA, d(A_dot)/dK
        [mu[1, 1], mu[2, 1]],  # d(K_dot)/dA, d(K_dot)/dK
    ], dtype=float)

    return b, J

def eig_and_times_from_mu(mu):
    b, J = affine_from_mu(mu)

    # fixed point (if possible)
    try:
        x_star = -np.linalg.solve(J, b)
    except np.linalg.LinAlgError:
        x_star = None  # singular J

    # eigen-decomposition of Jacobian
    evals, evecs = np.linalg.eig(J)  # evecs[:,i] corresponds to evals[i]

    # characteristic times
    taus = []
    periods = []
    for lam in evals:
        re = np.real(lam)
        im = np.imag(lam)

        if np.isclose(re, 0.0):
            tau = np.inf
        elif re < 0:
            tau = -1.0 / re   # stable residence time
        else:
            tau =  1.0 / re   # unstable escape time

        T = np.inf if np.isclose(im, 0.0) else (2*np.pi / abs(im))

        taus.append(tau)
        periods.append(T)

    return {
        "b": b,
        "J": J,
        "x_star": x_star,
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "taus": np.array(taus, dtype=float),
        "periods": np.array(periods, dtype=float),
    }