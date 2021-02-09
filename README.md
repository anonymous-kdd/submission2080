## This is the implementation of SIGN for the submission 2080 in KDD21.

### Dependencies

- python >= 3.6
- torch >= 1.4.0
- dgl == 0.4.3
- openbabel == 3.1.1 (optional, only for feature generation)
- tfbio >= 0.3 (optional, only for feature generation)

### Datasets
The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).
The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org).

Alternatively, we also created an anonymous dropbox account for downloading PDBbind and CSAR-HiQ datasets from the [link](https://www.dropbox.com/sh/xuwsl38u08zd44c/AADugOdpImRRfOj_fCDASe4Ra).

The downloaded data should be preprocessed to generate the protein-ligand graph and features:
```
python Feature_gen.py --data_path_core YOUR_DATASET_PATH --data_path_refined YOUR_DATASET_PATH --dataset_name refined --output_path YOUR_OUTPUT_PATH --cutoff 5
```
PS: cutoff is the threshold of cutoff distance between atoms.

Then run the preprocessing code for SIGN model:
```
python Preprocess.py --DATADIR YOUR_DATAS_PATH --dataset refined/general --target_dir YOUR_OUTPUT_PATH --cut_r 5 --n_space 6
```

Note that we provide the preprocessed testing files (both PDBbind core set and CSAR-HiQ set) in the anonymous [shared link](https://www.dropbox.com/sh/giiyc0zqa9okitb/AADgt23bsEhUNSh924O1fV_7a) to reproduce the experimental results in our paper. Before runing the testing script, please put the downloaded two files into the directory (./data/processed/).

### How to run
To train the model:
```
python main.py --cuda YOUR_DEVICE --model_dir MODEL_PATH_TO_SAVE --cut_r 5 --n_space 6
```
We also provide the pretrained model in the directory (./output/pretrained_model/) and the tesring script for reproducibility:
```
bash run_testing.sh
```
