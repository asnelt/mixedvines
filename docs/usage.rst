Usage
=====


Suppose that data are given in a NumPy array ``samples`` with shape ``(n, d)``,
where ``n`` is the number of samples and ``d`` is the number of elements.
First, specify which of the elements are continuous.  If, for instance, the
distribution has three elements and the first and last element are continuous
whereas the second element is discrete:

.. code-block:: python

    import numpy as np
    is_continuous = np.full((3), True, dtype=bool)
    is_continuous[1] = False

To fit a mixed vine to the samples:

.. code-block:: python

    from mixedvine import MixedVine
    vine = MixedVine.fit(samples, is_continuous)

``vine`` is now a ``MixedVine`` object.  To draw samples from the distribution,
calculate their density and estimate the distribution entropy in units of bits:

.. code-block:: python

    samples = vine.rvs(size=100)
    logpdf = vine.logpdf(samples)
    (entropy, standard_error_mean) = vine.entropy(sem_tol=1e-2)

Note that for the canonical vine, the order of elements is important.  Elements
should be sorted according to the importance of their dependencies to other
elements where elements with important dependencies to many other elements
should come first.  A heuristic way to select the order of elements is to
calculate Kendall's tau between all element pairs
(see ``scipy.stats.kendalltau``), to obtain a score for each element by summing
the tau's of the pairs the element occurs in and to sort elements in descending
order according to their scores.
