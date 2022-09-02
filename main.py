# example of how the classifier might be trained, then used

import argparse
from omegaconf import OmegaConf
from data_utils import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from pr_utils import MLP

def main():
    # get the settings for the script (stored in Configs/example_config.yml)
    parser = argparse.ArgumentParser(description="Single sample training example")
    parser.add_argument('--config', type=str, default="configs/example_config.yml")
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)

    dataset = Dataset(args)
    dataset.prepare_dataset()

    # we ~should~ be taking the data in blocks ( like first 60% of motions as train block )
    # this is just a quick way to make sure classes are equally represented as PoC.
    dataset.shuffle()

    train_dataset = dataset.split(args.train_val_test, 'train')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=default_collate)
    val_dataset   = dataset.split(args.train_val_test, 'val')
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=True, collate_fn=default_collate)
    test_dataset  = dataset.split(args.train_val_test, 'test')
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, collate_fn=default_collate)
    

    ## get model ready
    num_inputs=0
    if "EMG" in args.modalities:
        num_inputs += train_dataset.emg_data.shape[1]
    if "IMU" in args.modalities:
        num_inputs += train_dataset.imu_data.shape[1]
    num_outputs = len(train_dataset.classes)

    model = MLP(num_inputs, num_outputs, args)

    model.fit(train_dataloader, val_dataloader)

    model.test(test_dataloader)

if __name__ == "__main__":
    main()