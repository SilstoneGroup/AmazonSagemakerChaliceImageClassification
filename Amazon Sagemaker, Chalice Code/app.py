from chalice import Chalice
from chalice import BadRequestError
import boto3, json
import os
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import base64


app = Chalice(app_name='Tuwi')
sm = boto3.client('sagemaker')
s3 = boto3.client('s3')
app.debug = True


@app.route('/', methods=['POST'])
def index():
    body = app.current_request.json_body
    return {'ResponseMetadata': str(body['data'])}


@app.route('/train', methods=['POST'])
def train_job_by_name():
    job = sm.describe_training_job(TrainingJobName="image-classification-2019-05-30-09-23-34-189")

    body = app.current_request.json_body
    if 'TrainingJobName' not in body:
        raise BadRequestError('Missing new job name')
    else:
       job['TrainingJobName'] = body['TrainingJobName']
    if 'S3OutputPath' in body:
        job['OutputDataConfig']['S3OutputPath'] = body['S3OutputPath']

    if 'S3InputPathTraining' in body:
        job['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri'] = body['S3InputPathTraining']
    if 'S3InputPathValidation' in body:
        job['InputDataConfig'][1]['DataSource']['S3DataSource']['S3Uri'] = body['S3InputPathValidation']

    if 'S3InputPathTrainingLst' in body:
        job['InputDataConfig'][2]['DataSource']['S3DataSource']['S3Uri'] = body['S3InputPathTrainingLst']
    if 'S3InputPathValidationLst' in body:
        job['InputDataConfig'][3]['DataSource']['S3DataSource']['S3Uri'] = body['S3InputPathValidationLst']

    if 'num_classes' in body:
        job['HyperParameters']['num_classes'] = body['num_classes']
    if 'num_training_samples' in body:
        job['HyperParameters']['num_training_samples'] = body['num_training_samples']
    if 'mini_batch_size' in body:
        job['HyperParameters']['mini_batch_size'] = body['mini_batch_size']
    if 'resize' in body:
        job['HyperParameters']['resize'] = body['resize']

    return {'ResponseMetadata': str(job)}
    if 'VpcConfig' in job:
        resp = sm.create_training_job(
            TrainingJobName=job['TrainingJobName'], AlgorithmSpecification=job['AlgorithmSpecification'], RoleArn=body['RoleArn'],
            InputDataConfig=job['InputDataConfig'], OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            VpcConfig=job['VpcConfig'],
            Tags=job['Tags'] if 'Tags' in job else [])
    else:
        Because VpcConfig cannot be empty like HyperParameters or Tags :-/
        resp = sm.create_training_job(
            TrainingJobName=job['TrainingJobName'], AlgorithmSpecification=job['AlgorithmSpecification'], RoleArn=body['RoleArn'],
            InputDataConfig=job['InputDataConfig'], OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            Tags=job['Tags'] if 'Tags' in job else [])
        return {'ResponseMetadata': resp['ResponseMetadata']}


@app.route('/create/{modelname}/{trainingjobname}', methods=['POST'])
def create_model(modelname, trainingjobname):
    role = "arn:aws:iam::204716791960:role/service-role/AmazonSageMaker-ExecutionRole-20190416T014965"
    model_name = modelname
    training_job_name = trainingjobname

    info = sm.describe_training_job(TrainingJobName=training_job_name)
    model_data = info['ModelArtifacts']['S3ModelArtifacts']
    container = "685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest"

    primary_container = {
        'Image': container,
        'ModelDataUrl': model_data
    }

    create_model_response = sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=primary_container)

    model = create_model_response['ModelArn']

    endpoint_config_name = model_name + "endpointConfig"
    endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'}])

    endpoint_name = model_name + 'Endpoint'
    endpoint_params = {
        'EndpointName': endpoint_name,
        'EndpointConfigName': endpoint_config_name,
    }
    endpoint_response = sm.create_endpoint(**endpoint_params)

    response = sm.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    return {'Endpoint Created, Status:' + status}

    wait until the status has changed
    sm.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    
    # print the status of the endpoint
    endpoint_response = sm.describe_endpoint(EndpointName=endpoint_name)
    status = endpoint_response['EndpointStatus']
    return {'Endpoint Created'}
    
    if status != 'InService':
        return {'Endpoint creation failed'}





@app.route('/image', methods=['POST'])
def imagetry():
    # dictionary of the transformations we defined earlier
    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }
    folder_path = 'images/cat'
    num_files_desired = 10
    # find all files paths from the folder
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # return {'ResponseMetadata': str(images)}
    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:

            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
            new_file_path = '%s/aug_image_%s.jpg' % (folder_path, num_generated_files)
            return {'ResponseMetadata': "uploaded"}
            # uploadOnS3(transformed_image)
            # write image to the disk
            io.imsave(new_file_path, transformed_image)
        num_generated_files += 1


def uploadOnS3():
    uploadedfile = s3.meta.client.upload_file(Filename='air.jpg', Bucket='testdatauploads', Key='testimage.png')
    return uploadedfile


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]