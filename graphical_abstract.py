import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
from scipy.stats import gaussian_kde
from matplotlib import rcParams
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

def _nearest_idx(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr) - float(v))))

def _within_bounds(A, K, a_min=0, a_max=7, k_min=0, k_max=10):
    A = np.asarray(A); K = np.asarray(K)
    return (A >= a_min) & (A <= a_max) & (K >= k_min) & (K <= k_max)

def _preop_points(df):
    if "Patient_ID" in df.columns:
        firsts = (df.sort_values("Years_mean")
                    .drop_duplicates("Patient_ID", keep="first"))
    else:
        tmin = df["Years_mean"].min()
        firsts = df[df["Years_mean"] == tmin]
    A0 = firsts["SurfaceArea_Norm_mean"].to_numpy()
    K0 = firsts["IntGaussian_Fluct_Norm_mean"].to_numpy()
    m = _within_bounds(A0, K0)
    return np.c_[A0[m], K0[m]]

def _clip_trajs(trajs, a_min=0, a_max=7, k_min=0, k_max=10):
    trajs = np.array(trajs, copy=True)
    trajs[:, :, 0] = np.clip(trajs[:, :, 0], a_min, a_max)
    trajs[:, :, 1] = np.clip(trajs[:, :, 1], k_min, k_max)
    return trajs

def _sample_ensembles(res, n_traj=100, t_eval=None, x0_from_data=True, x0_jitter=None, rng=None):
    """
    Draw n_traj trajectories using coefficient/posterior sampling.
    Initial points are taken from pre-op *within bounds*; trajectories are
    simulated then clipped to the A/K box.

    res must contain: 'model', 'coef', 'sd', 'df', 'x0', 'sol'
    model.simulate(x0, t_eval, coef_draw) -> (T,2) with columns (A, dK).
    """
    rng = np.random.default_rng() if rng is None else rng
    model = res["model"]
    coef  = res["coef"]
    sd    = res["sd"]
    df    = res["df"].copy()

    if t_eval is None:
        t_eval = res["sol"].t

    # initial conditions
    if x0_from_data:
        x0_pool = _preop_points(df)   # already bounded
        if len(x0_pool) == 0:
            X0 = np.repeat(res["x0"][None, :], n_traj, axis=0)
        else:
            ids = rng.integers(0, len(x0_pool), size=n_traj)
            X0 = x0_pool[ids]
    else:
        X0 = np.repeat(res["x0"][None, :], n_traj, axis=0)

    if x0_jitter is not None:
        X0 = X0 + rng.normal(0.0, x0_jitter, size=X0.shape)
        X0[:, 0] = np.clip(X0[:, 0], 0, 7)
        X0[:, 1] = np.clip(X0[:, 1], 0, 10)

    nd, nf = coef.shape
    Cdraws = rng.normal(coef, sd, size=(n_traj, nd, nf))

    trajs = []
    for k in range(n_traj):
        Y = model.simulate(X0[k], t_eval, Cdraws[k])  # (T,2) = (A, dK)
        trajs.append(Y)
    trajs = np.stack(trajs, axis=0)
    trajs = _clip_trajs(trajs)
    return X0, trajs, t_eval

def _kde_contour_AK_plane(
    ax3d,
    A, K,
    t_slice,
    *,
    levels=(0.5, 0.8),
    bounds=(0, 7, 0, 10),
    color='k',
    lw=2.0,
    alpha=1.0,
    ngrid=50,
):
    """
    Draw 2D KDE contours in (A, dK) and embed them as 3D lines on the
    plane x = t_slice (time), with y = A, z = dK.

    - ax3d: 3D axis
    - A, K: 1D arrays of coordinates
    - t_slice: scalar time (x-coordinate of the plane)
    - levels: contour probability levels (in [0,1] after normalization)
    - bounds: (a_min, a_max, k_min, k_max) for (A, dK)
    """
    A = np.asarray(A)
    K = np.asarray(K)

    a_min, a_max, k_min, k_max = bounds
    m = _within_bounds(A, K, a_min, a_max, k_min, k_max)
    A = A[m]; K = K[m]
    if A.size < 5:
        return

    # 2D grid in (A, dK)
    Ay, Ky = np.meshgrid(
        np.linspace(a_min, a_max, ngrid),
        np.linspace(k_min, k_max, ngrid)
    )

    pts = np.vstack([A, K])
    kde = gaussian_kde(pts)
    Z = kde(np.vstack([Ay.ravel(), Ky.ravel()])).reshape(Ay.shape)

    # normalize so "levels" are in [0,1]
    Z = Z / (Z.max() + 1e-12)

    # --- build 2D contours in (A, dK) space using a throwaway figure ---
    fig2, ax2 = plt.subplots()
    cs = ax2.contour(Ay, Ky, Z, levels=levels, colors=color)
    plt.close(fig2)

    # cs.allsegs is a list per level; each element is a list of (N_i, 2) arrays
    for lvl_segs in cs.allsegs:
        for seg in lvl_segs:
            # seg[:,0] = A, seg[:,1] = dK in 2D
            A_path = seg[:, 0]
            K_path = seg[:, 1]
            T_path = np.full_like(A_path, float(t_slice))

            line = art3d.Line3D(
                xs=T_path,
                ys=A_path,
                zs=K_path,
                color=color,
                lw=lw,
                alpha=alpha
            )
            ax3d.add_line(line)

def graphical_abstract(
    all_results,
    NonPath,
    n_traj_per_cohort=150,
    t_end=5.0,
    time_slices=(1.0, 2.5, 5.0),
    colors=('tab:blue', 'tab:red'),
    elev=15,
    azim=-60,
    dpi=100
):
    res_r, res_s = all_results
    t_eval = np.linspace(0, t_end, 450)
    X0_r, Traj_r, T = _sample_ensembles(res_r, n_traj=n_traj_per_cohort, t_eval=t_eval)
    X0_s, Traj_s, _ = _sample_ensembles(res_s, n_traj=n_traj_per_cohort, t_eval=t_eval)
    pre_r = _preop_points(res_r["df"])
    pre_s = _preop_points(res_s["df"])
    a_min, a_max = 0, 7
    k_min, k_max = 0, 10
    bounds = (a_min, a_max, k_min, k_max)

    fig = plt.figure(figsize=(10.5, 8.0), dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d', facecolor='white')

    if len(pre_r):
        ax.scatter(
            np.zeros(len(pre_r)),
            pre_r[:, 0],
            pre_r[:, 1],
            s=25, c=colors[0], edgecolors='k', linewidths=0.3, depthshade=False
        )
    if len(pre_s):
        ax.scatter(
            np.zeros(len(pre_s)),
            pre_s[:, 0],
            pre_s[:, 1],
            s=25, c=colors[1], edgecolors='k', linewidths=0.3, depthshade=False
        )

    try:
        A_g = np.asarray(NonPath['SurfaceArea_Norm_mean'])
        K_g = np.asarray(NonPath['IntGaussian_Fluct_Norm_mean'])
        m_g = _within_bounds(A_g, K_g, a_min, a_max, k_min, k_max)
        if m_g.any():
            ax.scatter(
                np.zeros(m_g.sum()),
                A_g[m_g],
                K_g[m_g],
                s=25, c='grey', edgecolors='k', linewidths=0.3, depthshade=False
            )
    except Exception:
        pass

    for Y in Traj_r:
        ax.plot(T, Y[:,0], Y[:,1], color=colors[0], alpha=0.1, lw=1)
    for Y in Traj_s:
        ax.plot(T, Y[:,0], Y[:,1], color=colors[1], alpha=0.1, lw=1)

    a_min, a_max = 0, 7
    k_min, k_max = 0, 10
    bounds = (a_min, a_max, k_min, k_max)

    levels = (0.5, 0.8)

    for ts in time_slices:
        idx = _nearest_idx(T, ts)

        A_r = Traj_r[:, idx, 0]
        K_r = Traj_r[:, idx, 1]
        _kde_contour_AK_plane(
            ax, A_r, K_r,
            t_slice=ts,
            levels=levels,
            bounds=bounds,
            color=colors[0],
            lw=2.0,
            alpha=0.95
        )

        A_s = Traj_s[:, idx, 0]
        K_s = Traj_s[:, idx, 1]
        _kde_contour_AK_plane(
            ax, A_s, K_s,
            t_slice=ts,
            levels=levels,
            bounds=bounds,
            color=colors[1],
            lw=2.0,
            alpha=0.95
        )

    try:
        A_g = np.asarray(NonPath['SurfaceArea_Norm_mean'])
        K_g = np.asarray(NonPath['IntGaussian_Fluct_Norm_mean'])
        t_grey = 5.0

        _kde_contour_AK_plane(
            ax,
            A_g, K_g,
            t_slice=1,
            levels=(0.1, 0.50),
            bounds=bounds,
            color='grey',
            lw=2,
            alpha=0.75,
        )

        _kde_contour_AK_plane(
            ax,
            A_g, K_g,
            t_slice=2.5,
            levels=(0.1, 0.50),
            bounds=bounds,
            color='grey',
            lw=2,
            alpha=0.75,
        )

        _kde_contour_AK_plane(
            ax,
            A_g, K_g,
            t_slice=t_grey,
            levels=(0.1, 0.50),
            bounds=bounds,
            color='grey',
            lw=2.5,
            alpha=0.95,
        )
    except Exception:
        pass

    def time_plane(x0, color='gray', alpha=0.07):
        verts = [
            (x0, a_min, k_min),
            (x0, a_max, k_min),
            (x0, a_max, k_max),
            (x0, a_min, k_max)
        ]
        poly = Poly3DCollection([verts], facecolors=[color], alpha=alpha)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

    for ts in time_slices:
        time_plane(ts)

    ax.set_xlabel(r'Time', labelpad=20)
    ax.set_ylabel(r'Aortic Size', labelpad=12)
    ax.set_zlabel(r'Aortic Shape', labelpad=12)  

    ax.set_xlim(0, t_end)
    ax.set_xticks([0, 1, 2.5, 5])
    ax.set_xticklabels(['EVAR', 'CT 1', 'CT 2', 'CT 3']) 
    ax.set_ylim(a_min, a_max)
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels(['2', '4', '6']) 
    ax.set_zlim(k_min, k_max)
    ax.set_zticks([2, 6, 10])
    ax.set_zticklabels(['2', '6', '10'])
    ax.view_init(elev=elev, azim=azim)

    ax.grid(False)
    plt.tight_layout()
    return fig, ax, Traj_r, Traj_s

def style_3d_axis_tnr(ax, label_fs=20, tick_fs=16):
    """
    Make a 3D axis use Times New Roman with larger labels and ticks.
    Call this AFTER the 3D plot is created.
    """
    # Axis labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs, fontname="Times New Roman", labelpad=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fs, fontname="Times New Roman", labelpad=14)
    ax.set_zlabel(ax.get_zlabel(), fontsize=label_fs, fontname="Times New Roman", labelpad=14)

    # Tick labels
    for lab in ax.get_xticklabels():
        lab.set_fontname("Times New Roman")
        lab.set_fontsize(tick_fs)
    for lab in ax.get_yticklabels():
        lab.set_fontname("Times New Roman")
        lab.set_fontsize(tick_fs)
    for lab in ax.get_zticklabels():
        lab.set_fontname("Times New Roman")
        lab.set_fontsize(tick_fs)

    # Title (if any)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=label_fs+2, fontname="Times New Roman", pad=12)