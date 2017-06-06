**ADMML**: Alternating Direction Method based Scalable Machine Learning on Spark
================================================================================

The advent of big-data has seen an emergence of research on scalable machine learning (ML) algorithms and big data platforms. Several software frameworks have been introduced to handle the data deluge like, MapReduce, Hadoop, and Spark etc. Among them, Spark has been widely used by the ML community. Spark supports distributed in memory computations and provides practitioners with a powerful, fast, scalable and easy way to build ML algorithms. Although there have been several Spark based ML libraries, there are very few packages that cover a wide range of problems with fast and accurate results. This tool provides an Alternating Direction Method of Multipliers (ADMM) based approach that can be used as a general framework to accurately solve several standard and variants of most widely used machine learning algorithms at scale.

The precursor to this tool is presented in [1,2]. It majorly follows the work provided in [3]. For implementation details please cite [1].

1. Dhar S, Yi C, Ramakrishnan N, Shah M. `ADMM based scalable machine learning on Spark.` InBig Data (Big Data), 2015 IEEE International Conference on 2015 Oct 29 (pp. 1174-1182). IEEE.
2. Kamath G, Dhar S, Ramakrishnan N, Hallac D, Leskovec J, Shah M.`Scalable Machine Learning on Spark for multiclass problems`, Baylearn 2016
3. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. `Distributed optimization and statistical learning via the alternating direction method of multipliers.` Foundations and Trends in Machine Learning. 2011 Jan 1;3(1):1-22.

Presented in Spark Summit 2017. ( https://spark-summit.org/2017/events/admm-based-scalable-machine-learning-on-apache-spark/ )


Installation and Configuration
------------------------------

Dependencies:

* `Apache Spark <https://github.com/apache/spark>`_ (Need Apache Spark 2.0.2 or higher. Tested on version 2.0.2)
* `NumPy <http://www.numpy.org/>`_ (tested on version 1.10.4)
* `setuptools <https://github.com/pypa/setuptools>`_ (tested on version 34.4.1)

(These instructions were tested on centos2.6 only, but they should work on other platforms. Additionally, it has been tested on Python 2.7).

1. Unzip the downloaded zipped file.
2. Navigate to the zipped <ADMML> folder, i.e. the folder which contains setup.py
3. Build the .egg files (``python setup.py bdist_egg``)
4. Launch Pyspark and distribute the .egg file to all the cluster nodes for the pyspark context:
   ``sc.addPyFile('<ADMML folder absolute path>/dist/admml-0.1-py2.7.egg')``


Try the following example regression code:

.. code-block:: python

	import admml.utils as utils
	import admml.mlalgs as ml
	trndata = sc.textFile('<URI of a CSV data>')
	trndata = trndata.map(utils.parse_vector)
	algoParam = ml.setDefaultAlgoParams()
	w, output, fval = ml.ADMMLeastSquares(trndata,algoParam)



Building the Documentation
--------------------------

(These instructions were tested on Windows only, but they should work on other platforms.)

In order to build the documentation, you will need the following Python packages:

* `Sphinx <https://pypi.python.org/pypi/Sphinx>`_ (tested on version 1.5.5)
* `numpydoc <https://pypi.python.org/pypi/numpydoc>`_ (tested on version 0.6.0)
* `sphinx_rtd_theme <https://pypi.python.org/pypi/sphinx_rtd_theme>`_ (tested on version 0.2.4)
* `mock <https://pypi.python.org/pypi/mock>`_ (tested on version 2.0.0)

With these prerequisites in place, the documentation can be built as follows.

1. Navigate to the ``ADMML/docs`` folder.
2. Run the command ``sphinx-apidoc -f -e -o source/ ../admml/``
3. Run the command ``make html`` (repeat this step until there are no warnings, which should require no more than 3 runs)


License
--------
ADMML is open-sourced under the Apache-2.0 license. See the `LICENSE <LICENSE>`_ file for details.

For a list of other open source components included in ADMML, see the
file `3rd-party-licenses.txt <3rd-party-licenses.txt>`_


Contact
-------
Sauptik Dhar <sauptik.dhar@us.bosch.com>
