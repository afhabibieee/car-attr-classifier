import random, os, json, argparse
from configs import CAR_SPECS_DIR

def names_roots(cars):
    
    return {
        'class_names': cars,
        'class_roots': [os.path.join(CAR_SPECS_DIR, 'Front', class_name) for class_name in cars]
    }

def jsonfile(train_size, seed):
    random.seed(seed)
    
    cars = sorted(os.listdir(os.path.join(CAR_SPECS_DIR, 'Front')))
    
    n_train_class = int(round(train_size * len(cars)))
    train_cars = random.sample(cars, k=n_train_class)
    
    val_test = list(set(cars).difference(train_cars))
    val_cars = val_test[:round(len(val_test)/2)]
    test_cars = val_test[round(len(val_test)/2):]
    
    with open(os.path.join(CAR_SPECS_DIR, 'train.json'), 'w') as outfile:
        json.dump(names_roots(train_cars), outfile, indent=4)
        
    with open(os.path.join(CAR_SPECS_DIR, 'val.json'), 'w') as outfile:
        json.dump(names_roots(val_cars), outfile, indent=4)
    
    with open(os.path.join(CAR_SPECS_DIR, 'test.json'), 'w') as outfile:
        json.dump(names_roots(test_cars), outfile, indent=4)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='split dataset for train val and test set')
    parser.add_argument('--train_size', type=float, default=0.8, help='percentage of the amount of training data')
    parser.add_argument('--seed', type=int, default=27, help='random seed')
    
    args = parser.parse_args()

    jsonfile(args.train_size, args.seed)