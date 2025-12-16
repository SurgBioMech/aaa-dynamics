import numpy as np
import pandas as pd
import zsindy as zsindy
from zsindy.dynamical_models import DynamicalSystem
from zsindy.ml_module import ZSindy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import seaborn as sns
from matplotlib import rcParams
rcParams['mathtext.fontset']='cm'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

def estimate_rho(x, xdot):
    n = x.shape[0]
    theta = np.hstack((np.ones((n, 1)), x))
    C = np.dot(theta.T, theta)
    V = np.dot(theta.T, xdot)
    S2 = np.diag([1,1,1])
    mu = np.linalg.solve(C, V)
    residuals = xdot - np.dot(theta, mu)
    rho = np.sqrt(np.sum(residuals**2) / (n * xdot.shape[1]))
    sigma = rho**2 * np.linalg.inv(C)
    return mu, sigma, rho

def plot_var_exp(lambdas, var_exps):
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, var_exps, marker='o', linestyle=':', color='k')
    plt.xlabel('Regularization Weight (λ)')
    plt.ylabel('Variance Explained ($R^2$)')
    plt.title('Variance Explained vs. Regularization Weight')
    plt.grid(True)
    plt.show()

def sindy_tuner(lambdas, rho, x, t, xdot): 
    variance_explained = []
    for lmbda in lambdas:
        zmodel = ZSindy(poly_degree=1, 
                        lmbda=lmbda, 
                        max_num_terms=3, 
                        rho=rho,
                        variable_names=['A', 'K'])
        zmodel.fit(x, t, xdot)
        z_xdot_pred = zmodel.predict()
        residuals = xdot - z_xdot_pred
        total_variance = np.sum((xdot - np.mean(xdot, axis=0))**2)
        unexplained_variance = np.sum(residuals**2)
        r2_score = 1 - (unexplained_variance / total_variance)
        variance_explained.append(r2_score)
    plot_var_exp(lambdas, variance_explained)

def get_fixed_points(mu):
    beta_0 = mu[0, :]
    A = mu[1:, :].T
    x_fixed = np.linalg.lstsq(A, -beta_0, rcond=None)[0]
    return x_fixed

def DiffEq(t, y, coef):
    A, K = y
    dAdt = (coef[0][0] + coef[0][1] * A + coef[0][2] * K)
    dKdt = (coef[1][0] + coef[1][1] * A + coef[1][2] * K)
    return [dAdt, dKdt]

def zsindy_coeff_distribution_plot(
    x, t, xdot, rho,
    n_splits=10, train_frac=0.8,
    lmbda=1e-2,
    poly_degree=1,
    max_num_terms=3,
    variable_names=('A','K'),
    random_state=0,
    point_color='k',
    segment_color='#555',
    fig_size=(9, 5),
    dpi=100,
):
    rng = np.random.default_rng(random_state)
    N = x.shape[0]

    # store per-split coefficients and variances: shape (splits, 2 dims, 3 terms)
    split_coefs = np.zeros((n_splits, 2, 3))
    split_vars  = np.zeros((n_splits, 2, 3))

    for s in range(n_splits):
        idx = rng.choice(N, size=int(np.floor(train_frac * N)), replace=False)
        x_tr, t_tr, xdot_tr = x[idx], t[idx], xdot[idx]

        z = ZSindy(poly_degree=poly_degree,
                   lmbda=float(lmbda),
                   max_num_terms=max_num_terms,
                   rho=rho,
                   variable_names=list(variable_names))
        z.fit(x_tr, t_tr, xdot_tr)

        C = z.coefficients()                 # (2, 3): [const, A, K] per eqn
        V = z.coefficients_variance()        # (2, 3)
        split_coefs[s] = C
        split_vars[s]  = V

    # aggregate across splits
    # terms: 0=const, 1=A, 2=K; dims: 0 -> dA/dt, 1 -> dK/dt
    means = split_coefs.mean(axis=0)
    avg_vars = split_vars.mean(axis=0)
    avg_stds = np.sqrt(avg_vars)

    # Build plotting scaffolding
    y_labels = [
        r"1", r"$\widetilde{A}$", r"$\widetilde{\delta K}$",
        r"1", r"$\widetilde{A}$", r"$\widetilde{\delta K}$"
    ]
    y_positions = np.arange(len(y_labels))[::-1]
    all_pts = split_coefs.reshape(-1)
    all_spans = (means.reshape(-1) - 2*avg_stds.reshape(-1),
                 means.reshape(-1) + 2*avg_stds.reshape(-1))
    x_min = min(all_pts.min(), all_spans[0].min())
    x_max = max(all_pts.max(), all_spans[1].max())
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    xlim = (x_min - pad, x_max + pad)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    def y_idx(dim, term):
        return int(len(y_labels) - 1 - (dim*3 + term))

    # Draw segments and points
    for dim in (0,1):
        for term in (0,1,2):
            y = y_idx(dim, term)
            m = means[dim, term]
            s = avg_stds[dim, term]
            ax.hlines(y, m - s, m + s, color='k', linewidth=9, zorder=1)
            ax.hlines(y, m - s, m + s, color=segment_color, linewidth=5, zorder=2)
            pts = split_coefs[:, dim, term]
            ax.scatter(pts, np.full_like(pts, y), s=22, color=point_color, edgecolors='k', zorder=3)

    # Axes formatting
    ax.set_xlim(-0.4, 0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("")
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    grp_centers = [4, 1]
    ax.axhline(2.5, color='k', linewidth=1.0, alpha=1.0)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(grp_centers)
    ax2.set_yticklabels([
        r"$\frac{d\widetilde{A}}{dt}$",
        r"$\frac{d\,\widetilde{\delta K}}{dt}$"
    ], fontname="Times New Roman", fontsize=12)
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('outward', 40))
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', length=0)
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', length=0, labelsize=20)
    fig.tight_layout()
    return fig, ax

def sindy_subplot_regr(sol, x, t, ta, xdot, z_xdot_pred, Afixed, Kfixed, size_color, shape_color, z_xpred_ensemble, dpi=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    scale = 1

    mean_traj = z_xpred_ensemble[:, :, 0].mean(axis=0)
    std_traj  = z_xpred_ensemble[:, :, 0].std(axis=0)
    min_traj = z_xpred_ensemble[:, :, 0].min(axis=0)
    max_traj = z_xpred_ensemble[:, :, 0].max(axis=0)

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('$\\widetilde{A}$', fontsize=20)

    ax1.fill_between(ta, min_traj, max_traj, color=size_color, alpha=0.1, zorder=0, label='Mean ± 1 SD')
    ax1.plot(ta, mean_traj, color=size_color, linestyle='-', linewidth=2, zorder=1, label='Mean')
    ax1.plot(ta, min_traj, color=size_color, linestyle=':', linewidth=1.5, zorder=1)
    ax1.plot(ta, max_traj, color=size_color, linestyle=':', linewidth=1.5, zorder=1)
    ax1.scatter(t, x[:,0], color=size_color, alpha=1, edgecolors='black', linewidth=1, label='Observed Data')
    ax1.axhline(Afixed, color=size_color, linestyle='--', label='Fixed Point')

    ax1.quiver(t, x[:,0], np.ones_like(t), xdot[:,0], angles='xy', scale_units='xy', zorder=10,
            scale=1, color=size_color, width=0.005, alpha=0.5, label='d$A$/dt at Δt=1')
    ax1.set_yticks((0, 2, 4, 6, 8, 10))
    ax1.set_ylim(-1, 10)
    ax1.set_xticks((0, 2, 4, 6))
    ax1.set_xlim(-1,7)
    ax1.grid(ls='--', lw=0.5)
    ax1.set_title('Regressing Size', color='k', fontsize=20)

    mean_traj = z_xpred_ensemble[:, :, 1].mean(axis=0)
    std_traj  = z_xpred_ensemble[:, :, 1].std(axis=0)
    min_traj = z_xpred_ensemble[:, :, 1].min(axis=0)
    max_traj = z_xpred_ensemble[:, :, 1].max(axis=0)

    ax2.set_xlabel('Time [yr]')
    ax2.set_ylabel('$\\widetilde{\\delta K}$', fontsize=20)

    ax2.fill_between(ta, min_traj, max_traj, color=shape_color, alpha=0.1, zorder=0, label='Mean ± 1 SD')
    ax2.plot(ta, mean_traj, color=shape_color, linestyle='-', linewidth=2, zorder=1, label='Mean')
    ax2.plot(ta, min_traj, color=shape_color, linestyle=':', linewidth=1.5, zorder=1)
    ax2.plot(ta, max_traj, color=shape_color, linestyle=':', linewidth=1.5, zorder=1)
    ax2.scatter(t, x[:,1], color=shape_color, alpha=1, edgecolors='black', linewidth=1, label='Observed Data')
    ax2.axhline(Kfixed, color=shape_color, linestyle='--', label='Fixed Point')

    ax2.quiver(t, x[:,1], np.ones_like(t), xdot[:,1], angles='xy', scale_units='xy', zorder=10,
            scale=1, color=shape_color, width=0.005, alpha=0.5, label='d${\\delta K}$/dt at Δt=1')
    ax2.set_ylim(-1, 20)
    ax2.set_yticks((0, 5, 10, 15, 20))
    ax2.set_xlim(-1, 7)
    ax2.set_xticks((0, 2, 4, 6))
    ax2.grid(ls='--', lw=0.5, zorder=0)
    ax2.set_title('Regressing Shape', color='k', fontsize=20)

    plt.tight_layout()
    plt.show()

def sindy_subplot_stable(sol, x, t, ta, xdot, z_xdot_pred, Afixed, Kfixed, size_color, shape_color, z_xpred_ensemble, dpi=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    scale = 1

    mean_traj = z_xpred_ensemble[:, :, 0].mean(axis=0)
    std_traj  = z_xpred_ensemble[:, :, 0].std(axis=0)
    min_traj = z_xpred_ensemble[:, :, 0].min(axis=0)
    max_traj = z_xpred_ensemble[:, :, 0].max(axis=0)

    ax1.set_xlabel('Time [yr]')
    ax1.set_ylabel('$\\widetilde{A}$', fontsize=20)

    ax1.fill_between(ta, min_traj, max_traj, color=size_color, alpha=0.1, zorder=0, label='Mean ± 1 SD')
    ax1.plot(ta, mean_traj, color=size_color, linestyle='-', linewidth=2, zorder=1, label='Mean')
    ax1.plot(ta, min_traj, color=size_color, linestyle=':', linewidth=1.5, zorder=1)
    ax1.plot(ta, max_traj, color=size_color, linestyle=':', linewidth=1.5, zorder=1)
    ax1.scatter(t, x[:,0], color=size_color, alpha=1, edgecolors='black', linewidth=1, label='Observed Data')
    ax1.axhline(Afixed, color=size_color, linestyle='--', label='Fixed Point')

    ax1.quiver(t, x[:,0], np.ones_like(t), xdot[:,0], angles='xy', scale_units='xy', zorder=10,
            scale=1, color=size_color, width=0.005, alpha=0.5, label='d$A$/dt at Δt=1')
    ax1.set_yticks((0, 2, 4, 6, 8, 10))
    ax1.set_ylim(-1, 10)
    ax1.set_xlim(-1,6)
    ax1.set_xticks((0, 2, 4, 6))
    ax1.grid(ls='--', lw=0.5)
    ax1.set_title('Stable Size', color='k', fontsize=20)

    mean_traj = z_xpred_ensemble[:, :, 1].mean(axis=0)
    std_traj  = z_xpred_ensemble[:, :, 1].std(axis=0)
    min_traj = z_xpred_ensemble[:, :, 1].min(axis=0)
    max_traj = z_xpred_ensemble[:, :, 1].max(axis=0)

    ax2.set_xlabel('Time [yr]')
    ax2.set_ylabel('$\\widetilde{\\delta K}$', fontsize=20)

    ax2.fill_between(ta, min_traj, max_traj, color=shape_color, alpha=0.1, zorder=0, label='Mean ± 1 SD')
    ax2.plot(ta, mean_traj, color=shape_color, linestyle='-', linewidth=2, zorder=1, label='Mean')
    ax2.plot(ta, min_traj, color=shape_color, linestyle=':', linewidth=1.5, zorder=1)
    ax2.plot(ta, max_traj, color=shape_color, linestyle=':', linewidth=1.5, zorder=1)
    ax2.scatter(t, x[:,1], color=shape_color, alpha=1, edgecolors='black', linewidth=1, label='Observed Data')
    ax2.axhline(Kfixed, color=shape_color, linestyle='--', label='Fixed Point')

    ax2.quiver(t, x[:,1], np.ones_like(t), xdot[:,1], angles='xy', scale_units='xy', zorder=10,
            scale=1, color=shape_color, width=0.005, alpha=0.5, label='d${\\delta K}$/dt at Δt=1')
    ax2.set_ylim(-1, 20)
    ax2.set_yticks((0, 5, 10, 15, 20))
    ax2.set_xlim(-1, 6)
    ax2.set_xticks((0, 2, 4, 6))
    ax2.grid(ls='--', lw=0.5, zorder=0)
    ax2.set_title('Stable Shape', color='k', fontsize=20)
    ax2.set_yticks((0, 5, 10, 15, 20))

    plt.tight_layout()
    plt.show()

def zsindy_pipeline(regr_data, 
                    stable_data,
                    colors,
                    poly_degree=1,
                    max_terms=3,
                    x0=None,
                    x0_std=None,
                    ens_trials=1000,
                    plot_bool=True):

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
            x     = df[['SurfaceArea_Norm_mean','IntGaussian_Fluct_Norm_mean']].values
            t     = df['Years_mean'].values
            xdot  = df[['dSurfaceArea_Norm_mean/dYears_mean',
                        'dIntGaussian_Fluct_Norm_mean/dYears_mean']].values
            x0_use= x0 if x0 is not None else x[0]
            mu, sigma, rho   = estimate_rho(x, xdot)
            A_fp, K_fp= get_fixed_points(mu)

            model = ZSindy(poly_degree=poly_degree,
                           lmbda=lam,
                           max_num_terms=max_terms,
                           rho=rho,
                           variable_names=['A','K'])
            model.fit(x, t, xdot)

            coef = model.coefficients()
            var  = model.coefficients_variance()
            sol  = solve_ivp(lambda tt,yy: DiffEq(tt, yy, coef),
                             (0, 10), x0_use,
                             t_eval=np.linspace(0,10,500))
            
            nd, nf = coef.shape
            if x0_std is None:
                x0_samps = np.tile(x0_use, (ens_trials,1))
            else:
                x0_samps = np.random.normal(x0_use, x0_std,
                                            size=(ens_trials,len(x0_use)))
            c_samps = np.random.normal(coef.ravel(),
                                       np.sqrt(model.coefficients_variance()).ravel(),
                                       size=(ens_trials, nd*nf)
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
                'sd':    np.sqrt(var),
                'A_fp':  A_fp,
                'K_fp':  K_fp,
                'model': model,
                'x0':    x0_use,
                'rho':   rho,
                'sigma': sigma,
                'mu':    mu,
                'B': mu[1:,:].T,
                'ens': ens
            })  

    names_regr   = [r['name'] for r in results_regr]
    names_stable = [s['name'] for s in results_stable]

    if plot_bool:
        fig1, axes1 = plt.subplots(N, 2,
                                figsize=(12, 4*N),
                                facecolor='whitesmoke',
                                squeeze=False)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        for i in range(N):
            axA, axK = axes1[i, 0], axes1[i, 1]
            axA.text(0.02, 0.95,
                    f"{names_regr[i]}",
                    transform=axA.transAxes,
                    va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.7))

            for results, color, tag in [
                (results_regr[i], colors['Regr'], 'regr'),
                (results_stable[i], colors['Stable'], 'stable')
            ]:
                name   = results['name']
                sol    = results['sol']
                x      = results['x']
                A_fp   = results['A_fp']
                K_fp   = results['K_fp']
                cA, cK = color, color

                # — Size (A) plot —
                axA.plot(sol.t, sol.y[0], '-', lw=3, color=cA,
                        label=f"{name} ({tag})")
                axA.scatter(results['t'], x[:,0], color=cA, s=20)
                axA.axhline(A_fp, color=cA, ls='--', lw=2)
                axA.set_title("Size")
                axA.set_ylabel('A')
                axA.set_ylim(0, 10)

                # — Shape (δK) plot —
                axK.plot(sol.t, sol.y[1], '-', lw=3, color=cK)
                axK.scatter(results['t'], x[:,1], color=cK, s=20)
                axK.axhline(K_fp, color=cK, ls='--', lw=2)
                axK.set_title("Shape")
                axK.set_ylabel('K')
                axK.set_ylim(0, 15)

            if i == 0:
                axA.legend(loc='upper right')
            axA.set_xlabel('Years')
            axK.set_xlabel('Years')

        plt.tight_layout()
        plt.show()

        fig2 = plt.figure(figsize=(18, 4*N), facecolor='whitesmoke')
        outer_gs = gridspec.GridSpec(N, 4, figure=fig2, hspace=0.5, wspace=0.3)

        for i in range(N):
            res_r = results_regr[i]
            res_s = results_stable[i]

            sol_r, coef_r, sd_r = res_r['sol'], res_r['coef'], res_r['sd']
            model_r, x0_r       = res_r['model'], res_r['x0']
            if x0_std is None:
                x0_samps_r = np.tile(x0_r, (ens_trials, 1))
            else:
                x0_samps_r = np.random.normal(x0_r, x0_std,
                                            size=(ens_trials, len(x0_r)))
            nd, nf = coef_r.shape
            c_samps_r = np.random.normal(coef_r.reshape(-1),
                                        sd_r.reshape(-1),
                                        size=(ens_trials, nd*nf)
                                        ).reshape(ens_trials, nd, nf)
            ens_r = np.stack([ model_r.simulate(x0_samps_r[k], sol_r.t,
                                                c_samps_r[k])
                            for k in range(ens_trials) ])

            sol_s, coef_s, sd_s = res_s['sol'], res_s['coef'], res_s['sd']
            model_s, x0_s       = res_s['model'], res_s['x0']
            if x0_std is None:
                x0_samps_s = np.tile(x0_s, (ens_trials, 1))
            else:
                x0_samps_s = np.random.normal(x0_s, x0_std,
                                            size=(ens_trials, len(x0_s)))
            nd, nf = coef_s.shape
            c_samps_s = np.random.normal(coef_s.reshape(-1),
                                        sd_s.reshape(-1),
                                        size=(ens_trials, nd*nf)
                                        ).reshape(ens_trials, nd, nf)
            ens_s = np.stack([ model_s.simulate(x0_samps_s[k], sol_s.t,
                                                c_samps_s[k])
                            for k in range(ens_trials) ])

            # — Panel 1: A ensemble —
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

            # — Panel 2: K ensemble —
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

            # — Panel 3: A distributions at t=0,2.5,5 —
            innerA = outer_gs[i, 2].subgridspec(3, 1, hspace=0.3)
            for k_idx, tp in enumerate([0, 2.5, 5]):
                ax = fig2.add_subplot(innerA[k_idx])
                idx = np.argmin(np.abs(sol_r.t - tp))
                sns.distplot(ens_r[:, idx, 0],
                            hist=False, kde=True,
                            color=colors['Regr'], ax=ax)
                sns.distplot(ens_s[:, idx, 0],
                            hist=False, kde=True,
                            color=colors['Stable'], ax=ax)
                ax.set_xlim(0, 10)
                ax.set_ylabel(f"t={tp}")
                ax.set_yticks([]); ax.set_yticklabels([])
                if k_idx < 2:
                    ax.set_xticks([]); ax.set_xticklabels([])
                if k_idx == 0:
                    ax.set_title("Size (A) distributions")

            # — Panel 4: K distributions at t=0,2.5,5 —
            innerK = outer_gs[i, 3].subgridspec(3, 1, hspace=0.3)
            for k_idx, tp in enumerate([0, 2.5, 5]):
                ax = fig2.add_subplot(innerK[k_idx])
                idx = np.argmin(np.abs(sol_r.t - tp))
                sns.distplot(ens_r[:, idx, 1],
                            hist=False, kde=True,
                            color=colors['Regr'], ax=ax)
                sns.distplot(ens_s[:, idx, 1],
                            hist=False, kde=True,
                            color=colors['Stable'], ax=ax)
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
                    'IntGaussian_Fluct_Norm_mean': ens[trial, j, 1]
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
                    'IntGaussian_Fluct_Norm_mean': ens[trial, j, 1]
                })
    stable_ens = pd.DataFrame(stable_rows)

    all_results = results_regr + results_stable
    return all_results, regr_ens, stable_ens