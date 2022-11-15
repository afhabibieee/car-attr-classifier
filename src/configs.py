from torch import device, cuda

CAR_SPECS_DIR = '../data/48_cars'

IMAGE_SIZE = 224

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

N_WAY = 5
N_SHOT = 5
N_QUERY = 10
N_WORKERS = 2
N_TASK = 200
N_TRAINING_EPISODES = 500
N_VALIDATION_TASK = 100

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = device('cuda' if cuda.is_available() else 'cpu')