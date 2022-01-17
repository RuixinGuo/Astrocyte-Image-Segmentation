There are 4 groups of train-predict programs:

+ **train.py** and **predict.py**: ensemble learning with 3 models.

+ **train5.py** and **predict5.py**: ensemble learning with 5 models.

+ **train7.py** and **predict7.py**: ensemble learning with 7 models.

+ **train10.py**: ensemble learning with 10 models. The ensemble model is trained with 2 GPUs, each of which trains 5 models. I didn't write predict10.py because I am lazy.

Before running **train*.py**, you need to create a directory named **model** and put all independently trained models to the directory.
