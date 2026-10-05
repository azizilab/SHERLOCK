Installation
============

SHERLOCK supports **Python 3.10 or newer** and runs on CPU or GPU. A GPU is
recommended for the larger screens.

Pip
---

.. code-block:: bash

   pip install sherlock-perturb

The distribution is called ``sherlock-perturb``; the package is imported as
``sherlock``:

.. code-block:: python

   import sherlock as slk

Install PyTorch first if you need a specific CUDA build
(see `pytorch.org <https://pytorch.org/get-started/locally/>`_); otherwise pip
installs the default wheel for your platform.

Optional extras
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - Adds
   * - ``tutorials``
     - Leiden clustering, UMAP of interaction metrics, gene-set enrichment, Jupyter
   * - ``benchmarks``
     - the baseline models (cVAE, sVAE, contrastiveVI and scGen through
       ``scvi-tools``; GEARS through ``cell-gears``)
   * - ``docs``
     - Sphinx and the theme used for this site
   * - ``all``
     - ``tutorials``, ``benchmarks`` and ``test``

.. code-block:: bash

   pip install "sherlock-perturb[tutorials]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/azizilab/SHERLOCK.git
   cd SHERLOCK
   pip install -e ".[tutorials]"

The repository folder is itself the ``sherlock`` package. The analysis notebooks
in ``notebooks/`` import the installed package, so install it this way before
running them.

Datasets
--------

The two public datasets used in the tutorials are downloaded on first use:

.. code-block:: python

   import sherlock as slk

   adata = slk.datasets.replogle()   # ~570 MB
   adata = slk.datasets.norman()     # ~1.7 GB

Files are stored in ``$SHERLOCK_DATA_DIR`` (default ``~/.cache/sherlock``). The
processed versions of the other datasets analysed in the manuscript are available
from the corresponding authors on request.

Conda
-----

.. code-block:: bash

   git clone https://github.com/azizilab/SHERLOCK.git
   cd SHERLOCK
   conda env create -f environment.yml
   conda activate sherlock

``environment.yml`` installs PyTorch with CUDA 12.1 and then SHERLOCK from the
checkout. On a CPU-only machine, replace ``pytorch-cuda=12.1`` with ``cpuonly``.

Verify the install
------------------

.. code-block:: bash

   python -c "import sherlock as slk; print(slk.__version__)"
   pytest tests    # from a source checkout, with the `test` extra

Using the tutorials in Jupyter
------------------------------

Register the environment you installed SHERLOCK into as a kernel and select it
when you open a tutorial:

.. code-block:: bash

   python -m ipykernel install --user --name sherlock --display-name "Python (SHERLOCK)"
