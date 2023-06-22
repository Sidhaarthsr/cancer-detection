*Cancer Detection*
=================
AI-driven Python Project for classifying Histopathological images for cancer detection

*Description*
^^^^^^^^^^^^
Inspired by the recent success of Artificial Intelligence in the field of Medical Imaging, we propose three different models to
classify Histopathology images into different types of lung and colon cancer. Specifically, we categorize each image into 5 classes - lung benign tissue, lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, and colon benign tissue 

The first two models are
decision trees in semi-supervised and fully-supervised
settings. Then we propose a CNN model in fully supervised settings


*Requirements*
^^^^^^^^^^^^^
To run the Python code in this project, the following libraries and dependencies are required.

**1. Pytorch** 

To install on windows with ``pip`` run this command::

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

For more information on different OS and configurations visit the official `website <https://pytorch.org/get-started/locally/>`_

**2. Scikit-Learn**

To install on windows with ``pip`` run this command::

   pip install -U scikit-learn

For different OS visit the official `website <https://scikit-learn.org/stable/install.html>`_

**3. Numpy**::

   pip install numpy

**4. OpenCV**::

   pip install opencv-contrib-python

**5. Scikit-image**::

   pip install scikit-image




*Training/Validation of the model*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. Supervised Decision Tree Model:**
  
Specify the ``dataset_path`` in the ``dtree_supervised.py`` file and then run the file.

**2. Semi-Supervised Decision Tree Model:**

Specify the ``dataset_path`` in the ``cancer_detection_dtree_semi_supervised.py`` file and then run the file.

**3. DNN Models:**

There are total 4 models available in the ``model.py``: Modelv1, Modelv2, Modelv3, Modelv4

To Train the model, specify the ``data_dir`` and ``model`` in the python file ``experiment.py``. 

Then, specify the ``data_dir`` in the python file ``data_loader.py``.

Then, run the file ``experiment.py``





*Source Code Packages*
^^^^^^^^^^^^^^^^^^^^^^^^^^

The project includes two source code packages: one in Scikit-learn and one in PyTorch. Both packages contain the necessary code for training and evaluating the model.

``scikit-learn`` :  `source code <https://github.com/scikit-learn/scikit-learn>`_

``pytorch`` :  `source code <https://github.com/pytorch/pytorch>`_


*Obtaining The Dataset*
^^^^^^^^^^^^^^^^^^^^^^^^^^

This dataset is available on `kaggle <https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images>`_
under creative commons license.


*Acknowledgments*
^^^^^^^^^^^^^^^^^^^^^^
Contributions:

`Shail Shah <https://github.com/shail2512-lm10>`_, `Atif Bilal <https://github.com/imatif17>`_, 
`Siddharth <https://github.com/Sidhaarthsr>`_, `Raj <https://github.com/raj8421>`_,
`Arash Azarfar <https://github.com/arazarfar>`_, `Y A Joarder <https://github.com/yajoarder>`_,
`Soorena <https://github.com/soorena374>`_, `Farzhad <https://github.com/FzS92>`_,




   
