import matplotlib.pyplot as plt
import numpy as np

def set_size(width=505.89, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_roc_curve(fpr, tpr, tpr_std=None, model_name="", ax=None, xlim=[-0.0005, 0.003], ylim=[0.3, 1.0], roc_auc=None, linestyle="-", color=None, semilogx=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if roc_auc:
        label = f"{model_name} (AUC = {roc_auc:.6f})"
    else:
        label = model_name
    if semilogx:
        ax.semilogx(fpr, tpr, lw=2, label=label, linestyle=linestyle, color=color)
    else:
        ax.plot(fpr, tpr, lw=2, label=label, linestyle=linestyle, color=color)
    if tpr_std is not None:
        tprs_upper = np.minimum(tpr + tpr_std, 1)
        tprs_lower = tpr - tpr_std
        color = ax.lines[-1].get_color()
        color = color[:-2] + "33"
        ax.fill_between(fpr, tprs_lower, tprs_upper, alpha=.2, color=color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax