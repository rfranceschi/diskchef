import numpy as np


def example_lin():
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)

    a = Parameter(name="a", min=0.1, max=2, truth=1)
    b = Parameter(name="b", min=-2, max=5, truth=2)
    fitter = EMCEEFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = BruteForceFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads, n_points=100)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = UltraNestFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads, transform=rescale_linear)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    plt.show()


def example_sqr():
    x = np.linspace(1, 10, 100)
    y = sqr([1, 2, 0], x)

    a = Parameter(name="a", min=0.1, max=2, truth=1)
    b = Parameter(name="b", min=-2, max=5, truth=2)
    c = Parameter(name="c", min=-2, max=5, truth=0)
    params = [a, b, c]

    fitter = EMCEEFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = BruteForceFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads, n_points=40)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = UltraNestFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads,
                             # transform=rescale_sqr
                             )
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    threads = 1
    # example_lin()
    example_sqr()