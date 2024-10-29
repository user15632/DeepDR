Model.Metric
===========================

`Click here </en/latest/document/Model/MetricCode.html>`_ to view source code.


.. code-block:: python

    class Metric:

It contains functions ``Metric.MSE``, ``Metric.R2``, and ``Metric.PCC``.


Metric.MSE
--------

.. code-block:: python

    def MSE(real,
            predict):


It is used to calculate mean square error.

**PARAMETERS:**

* **real** - The real value.
* **predict** - The predicted value.

**OUTPUTS:**

* **mse** - The mean square error.


Metric.R2
--------


.. code-block:: python

    def R2(real,
           predict):


It is used to calculate r square.

**PARAMETERS:**

* **real** - The real value.
* **predict** - The predicted value.

**OUTPUTS:**

* **r2** - The r square.


Metric.PCC
--------


.. code-block:: python

    def PCC(real,
            predict):


It is used to calculate pearson correlation coefficient.

**PARAMETERS:**

* **real** - The real value.
* **predict** - The predicted value.

**OUTPUTS:**

* **pcc** - The pearson correlation coefficient.
