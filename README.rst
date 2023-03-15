.. image:: ./logo/ourLogo.svg
  :width: 110
  :height: 110
  :align: left
  
.. image:: https://img.shields.io/badge/Creators-D.%20Bykhovsky%2C%20A.%20Rudyak%2C%20N.%20Tochilvosky-blue
  :align: center
  
.. image:: https://img.shields.io/badge/Version-v1.0.0-green
  :align: center

.. image:: https://img.shields.io/badge/License-MIT-lightgreen
  :align: center
  
|
  
Generate_corr_sequence
=============

We've created a Python function capable of creating ``L`` sized vecor of samples, having a desired CDF function and ACC.

To work with it, the user needs to have a desired function

For that purpose we made helper functions:

#. ``findCoeff`` - find the d coefficients of a given distribution
#. ``integration_function`` - a function that integrated over the non-inversed CDF solution
#. ``find_ro_x`` - add roh_x explained # maybe change the function name?
#. ``findFilter`` - find the required filter for the desired ACF
#. ``get_ranked_sequence`` - rank match the sequence to get the desired ACF
#. ``drawDebugPlots`` - if something doesnt work this might give an insight with plots about whats going on

Summary
-------
   
A python package for generating a random autocorrelated sequence with user preffered autocorrelation function and cumulative distribution function.

Usage Examples
-----

How to use the function.

.. code-block:: python

    result = function_name(arg1, arg2)
    print(result)

Arguments
---------

- |distribution-type| ``dist_obj`` - A distribution object from scipy.stats, default is ``uniform``.
- |function-type| ``desiredACF`` - A desired ACF function with ``m`` as variable, default is ``linear function``.
- |int-type| ``L`` - Number of desired samples, default is ``2^20``.
- |int-type| ``seed`` - Number as input for the random number generator, default is ``100``.
- |bool-type| ``debug`` - Plots intermidiate graphs, also helps with visualization, default is ``False``.





Returns
-------

- ``return_type``: Description of the return value.

Version History
---------------

- Version X.X.X: Description of changes made in this version.

See Also
--------

Other related functions, classes or modules that may be useful to reference.

References
----------

Any references or resources used to develop the function or related to the function.

Contributors
------------

- `Dima Bykhovsky <https://github.com/bykhov>`_
- Contributor Name (https://github.com/contributor_username)
- `Alexander Rudyak <https://github.com/AlexRudyak>`_

License
-------

This project is licensed under the `MIT <./LICENSE.md>`_ license.

Examples
=============

Rayleigh distribution with exp*cos ACF
-------

.. code-block:: python

    # Example usage of the function with Rayleigh distribution and an autocorrelation function
    import scipy
    import numpy as np
    from scipy.stats import rayleigh

    m = np.arange(0, 100)
    desiredACF = np.exp(-0.05 * np.abs(m)) * np.cos(0.25 * np.abs(m))
    sequence = generate_corr_sequence(rayleigh, desiredACF=desiredACF, L=2 ** 20, seed=100, debug=True)
    
Probability Density Funciton before and after the ACF matching process
----
.. image:: ./examples/exp-0.05mcos0.25mpdf.png
  :align: center
  
AutoCorrelation Funciton before and after the ACF matching process
----
.. image:: ./examples/exp-0.05mcos0.25macf.png
  :align: center


.. |bool-type| image:: https://img.shields.io/badge/bool--x.svg?style=social
.. |int-type| image:: https://img.shields.io/badge/int--x.svg?style=social
.. |function-type| image:: https://img.shields.io/badge/function--x.svg?style=social
.. |distribution-type| image:: https://img.shields.io/badge/distribution--x.svg?style=social


