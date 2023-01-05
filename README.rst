============================
HFSS Bandstop filter matcher
============================
:Authors: - Joséphine Dupeyron Masini <josephine.masini@sorbonne-universite.fr>
          - Florian Dupeyron <florian.dupeyron@mugcat.fr>
:Date: October 2022

This scripts shows how to use of the PyAEDT_ library to match an HFSS design's :math:`S_{21}` response to an input
`.s2p` file.

In this example, two parameters, :code:`$Ers` and :code:`$tands` are matched against the input curve.

.. _PyAEDT: https://github.com/pyansys/pyaedt

How to use
==========

1. Make sure `PyAEDT`_ is installed: 

   .. code::
       
       pip install pyaedt

2. Configure input. See the :ref:`Input data` section.
3. Launch the `s2p_processor.py` script to launch processing:

    .. code::

        python s2p_processor.py <input_folder> <output_folder>

4. Output data should be available in the provided :code:`<output_folder>` argument. You can then add plots in the folders:

    .. code::

        python plotter.py <output_folder>


Input data
==========

TODO

