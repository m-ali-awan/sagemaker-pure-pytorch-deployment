# sagemaker-pure-pytorch-deployment

## Steps to Follow:



> AWS uses Docker containers for deployment, and for PyTorch we will do the same . Now **framework-version** is very important here, we will use 1.4.0. as 
> in the latest i.e 1.6.0, and onwards Torchserve is being used, which requires a particular .md file for serving. It can also be done, but it will require different procedure.


For followin **TorchServe** serve, following steps can be followed:
> https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/


### Steps To Follow:

#### 1.  First tar the exported model .pkl file:
```
path2=Path('TarredModels')
import tarfile
with tarfile.open(path2/'model.tar.gz', 'w:gz') as f:
    f.add('export.pkl', arcname='model.pkl')
```

#### 2. Uploading model to s3:
```
model_location = sess.upload_data(str(path2/'model.tar.gz'), key_prefix=prefix)
model_location
```

#### 3. Now we have to provide .py files, and requirements.txt in a directory

**The PyTorchModel class allows you to define an environment for making inference using your model artifact**

Once it is properly configured, it can be used to create a SageMaker endpoint on an EC2 instance. The SageMaker endpoint is a containerized environment that uses your trained model to make inference on incoming data via RESTful API calls.

Some common parameters used to initiate the PyTorchModel class are: - entry_point: A user defined python file to be used by the inference image as handlers of incoming requests - source_dir: The directory of the entry_point - role: An IAM role to make AWS service requests - model_data: the S3 location of the compressed model artifact. It can be a path to a local file if the endpoint is to be deployed on the SageMaker instance you are using to run this notebook (local mode) - framework_version: version of the PyTorch package to be used - py_version: python version to be used


```
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(model_data=model_location,
                     role=role,
                     framework_version='1.4.0',
                     py_version='py3',
                     entry_point='serve.py', 
                     source_dir='scripts',
                    predictor_cls=ImagePredictor)
```


#### 4. Entry Point, and Source dir:
We can specify the required packages in requirements.txt in **source_dir**, as usually there are version conflicts. These can be debugged by using **local** model if deployment, and then running docker container interactively to see which things are erroring out.


#### 5. Serialization, and Deserialization:

**Now as the image will have to be given in raw form, we have to include the target extensions in the formats which will be treated as binary in API Gateway**


Official documentation can be found here:
> https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
