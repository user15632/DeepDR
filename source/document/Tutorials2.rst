How to prepare your own data
==================================

The following are detailed tutorials on how to prepare your own data.

.. note::

   If you use data integrated in the library, ignore this section.


Prepare your gene subset
--------

.. note::

   If you want to use the default gene subset or not use the gene subset, or use the cell feature integrated in the library, ignore this section.

You need to prepare a ``gbk`` encoding ``txt`` file.
Each row in the file should be a gene name,
The following is an example:

.. code-block:: text

   ARHGEF10L
   RNF11
   GTF2IP1


Prepare your cell data
--------

.. note::

   If you want to use the cell feature integrated in the library, ignore this section.

You need to prepare a ``csv`` file with column separators of ``,``.
The first column in the file should be the gene name, the first row should be the cell name.
The following is an example:

+--------+----------+---------+----------+
| Gene   | CAL120   | DMS114  | CAL51    |
+========+==========+=========+==========+
| A1BG   | 6.208447 | 5.02581 | 5.506955 |
+--------+----------+---------+----------+
| A1CF   | 2.981775 | 2.947547| 2.872071 |
+--------+----------+---------+----------+
| A2M    | 3.133883 | 3.335711| 3.287678 |
+--------+----------+---------+----------+


Prepare your response data
--------

.. note::

   If you want to use the response data integrated in the library, ignore this section.

You need to prepare a ``csv`` file with column separators of ``,``.
The first row is the header, the second row to the last row are cell-drug pairs,
where the first column is the cell name, the second column is the drug name, and the third column is the drug response.
The following is an example:

+---------+---------+----------+
| Cell    | Drug    | Response |
+=========+=========+==========+
| 1321N1  | AEW541  | 0.7124   |
+---------+---------+----------+
| 22Rv1   | AEW541  | 1.6723   |
+---------+---------+----------+
| 42-MG-BA| AEW541  | 1.1852   |
+---------+---------+----------+
