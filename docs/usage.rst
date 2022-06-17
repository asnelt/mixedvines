Usage
=====


Suppose that data are given in a NumPy array ``samples`` with shape
``(n, d)``, where ``n`` is the number of samples and ``d`` is the number of
elements per sample.  First, specify which of the elements are continuous.
If, for instance, the distribution has three elements and the first and last
elements are continuous whereas the second element is discrete:

.. code-block:: python

    is_continuous = [True, False, True]

To fit a mixed vine to the samples:

.. code-block:: python

    from mixedvines.mixedvine import MixedVine
    vine = MixedVine.fit(samples, is_continuous)

``vine`` is now a ``MixedVine`` object.  Note that for the canonical vine, the
order of elements is important.  Elements should be sorted according to the
importance of their dependencies to other elements, where elements with
important dependencies to many other elements should come first.  A heuristic
way to select the order of elements is to calculate Kendall's tau between all
element pairs, to obtain a score for each element by summing the taus of the
pairs the element occurs in and to sort elements in descending order according
to their scores.  This is what the ``MixedVine.fit`` method does internally by
default to construct an improved canonical vine tree.  This internal sorting
is used to construct the vine tree only, so the order of elements is not
changed in a user visible way.  To prevent this internal sorting, set the
``keep_order`` argument to ``True``.

To draw samples from the distribution, calculate their density and estimate
the distribution entropy in units of bits:

.. code-block:: python

    samples = vine.rvs(size=100)
    logpdf = vine.logpdf(samples)
    (entropy, standard_error_mean) = vine.entropy(sem_tol=1e-2)
