# RangerOptimizerTensorflow
This code is a combination of two TensorFlow addons optimizers. The code is mostly loaded from there but slightly modified.
## Use
This TensorFlow optimizer combines the two <code>tf.keras.optimizers</code> RAdam and Lookahead into one optimizer called Ranger. This can instantly be fed into the <code>tf.keras.model.fit</code> / <code>tf.keras.model.fit_generator</code> method to fit a model using this optimizer. All the setting of the hyper parameters can therefore be done in a single step. 
## Setup
The python files were created for python version 3.7, although it might also work for past or future versions.
To use this class, some python modules need to be installed first. Using <code>pip</code> the packages can be installed by either typing 
<code>pip install -r requirements.txt</code>
in terminal, if the requirements.txt file exists in the current working directory or by typing
<code>pip install tensorflow==2.0.0 tensorflow-addons==0.6.0</code>
into the terminal (!python and pip need to be installed first, the recommended version for pip is at least 19.3.1). The versions of the modules listed above were used at the time of the creation of these files but future versions of these modules might alos work. Another way to install these packages is by using <code>conda</code>.
## Code
For using the optimizer there are two options:
1. Put the code straight into a python file:<br/>
For that the code from the file [plain.py](plain.py) should be copied into the python file.
2. Importing the class from a different python file:<br/>
For that the file [module.py](module.py) should be inserted into the project folder in which the executed file lies and imported at the top of the executed file:<br/>
<code>from module import Ranger</code>
<!---->
In the following python code the following elements should be included:<br/>
```python
# load the required modules
import tensorflow.keras as k

# load training data, transforms, any other setup for model
[...]

# define the tf.keras model
model = k.model.Sequential()
[...] # using the model.add([...]) function new layers can be added to the model

# compile and fit the model with Ranger optimizer
optimizer = Ranger([...]) # feed arguments of Ranger optimizer into creation function
model.compile(optimizer, [...]) # add aditional parameters
model.fit([...]) / model.fit_generator([...]) # fit the model using array data or a generator

model.save('path/to/model/name.h5') # save the model (optional but sensible)
```
The recommended way of using this class is by importing it as a module because docstrings are provided to document the module. In the plain.py file the documentation is not present for shortening the code.
## Credits
https://arxiv.org/abs/1908.03265 (RAdam paper)<br/>
https://arxiv.org/abs/1907.08610 (Lookahead paper)<br/>
https://github.com/tensorflow/addons/blob/v0.6.0/tensorflow_addons/optimizers/lookahead.py#L25-L171 (TensorFlow implementation of Lookahead optimizer)<br/>
https://github.com/tensorflow/addons/blob/v0.6.0/tensorflow_addons/optimizers/rectified_adam.py#L25-L306 (TensorFlow implementation of RAdam)
