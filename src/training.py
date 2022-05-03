
from utils.common import read_config
from utils.data_mgmt import get_data
import argparse
from utils.model import create_model, save_model
import os

#Read and print configurations present
def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, Y_train),(X_test, Y_test),(X_val, Y_val)=get_data(validation_datasize)
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_val, Y_val)
    model_name = config["artifacts"]["model_name"]

#saving the model
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)


#To parse and supply the configurations
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)

