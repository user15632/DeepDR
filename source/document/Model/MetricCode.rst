Model.Metric
===========================

`Click here </document/Model/Metric.html>`_ to go back to the reference.


.. code-block:: python

    class Metric:
        @staticmethod
        def MSE(real, predict):
            return np.mean((real - predict) ** 2)

        @staticmethod
        def R2(real, predict):
            return r2_score(real, predict)

        @staticmethod
        def PCC(real, predict):
            return np.corrcoef(real, predict)[0][1]
