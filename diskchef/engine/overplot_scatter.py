import numpy as np


def _parse_input(xs):
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    return xs


def _get_fig_axes(fig, K):
    if not fig.axes:
        return fig.subplots(K, K)
    try:
        return np.array(fig.axes).reshape((K, K))
    except ValueError:
        raise ValueError(
            (
                "Provided figure has {0} axes, but data has "
                "dimensions K={1}"
            ).format(len(fig.axes), K)
        )


def overplot_scatter(fig, xs, **kwargs):
    """
    Overplot points on a figure generated by ``corner.corner``

    Parameters
    ----------
    fig : Figure
        The figure generated by a call to :func:`corner.corner`.

    xs : array_like[nsamples, ndim]
       The coordinates of the points to be plotted. This must have an ``ndim``
       that is compatible with the :func:`corner.corner` call that originally
       generated the figure.

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.scatter``
        method.

    """
    kwargs["marker"] = kwargs.pop("marker", ".")
    # kwargs["linestyle"] = kwargs.pop("linestyle", "none")
    xs = _parse_input(xs)
    K = len(xs)
    axes = _get_fig_axes(fig, K)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            axes[k2, k1].scatter(xs[k1], xs[k2], **kwargs)


def overplot_hexbin(fig, xs, C, gridsize=20, reduce_C_function=np.nanmax, **kwargs):
    """
    Overplot hexbin on a figure generated by ``corner.corner``

    Parameters
    ----------
    fig : Figure
        The figure generated by a call to :func:`corner.corner`.

    xs : array_like[nsamples, ndim]
       The coordinates of the points to be plotted. This must have an ``ndim``
       that is compatible with the :func:`corner.corner` call that originally
       generated the figure.

    **kwargs
        Any remaining keyword arguments are passed to the ``ax.hexbin``
        method.

    """
    xs = _parse_input(xs)
    K = len(xs)
    axes = _get_fig_axes(fig, K)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            axes[k2, k1].hexbin(xs[k1], xs[k2], C, gridsize=gridsize, reduce_C_function=reduce_C_function, **kwargs)
