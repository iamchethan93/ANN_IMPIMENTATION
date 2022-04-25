from src.utils.common import read_config
from src.utils.data_mgmt import get_data
import argparse


#Read and print configurations present
def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["Params"]["validation_datasize"]
    (X_train, Y_train),(X_test, Y_test),(X_val, Y_val)=get_data(validation_datasize)

#To parse and supply the configurations
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c",default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)

