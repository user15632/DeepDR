DeepDR installation via pip
==================================

The following are detailed tutorials on how to install DeepDR.

Installation of DeepDR involves simply typing as follows:

.. code-block:: python

    pip install deepdr -i https://pypi.org/simple


Dependency libraries such as Pytorch and PyG will be installed automatically.

If you have problems with usage, try using the recommended Pytorch and PyG versions as follows:

.. code-block:: python

    conda create -n deepdr python=3.7.11
    conda activate deepdr
    pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch_geometric==2.0.3
    pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
    pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
    pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
    pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
    pip install deepdr -i https://pypi.org/simple
