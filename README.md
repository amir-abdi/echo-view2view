# Echo View Converter with Clinically Valid Constraints   

To initiate training, run 

     python3 src/main.py --dataset_path=$DATASETS/CAMUS --config=configs/config_2CH_4CH.json
     
The environment variable `$DATASET` is assumed to be set to 
where the CAMUS dataset directory is stored. 


- Unified all modes of the codes.
Now segmentation, patchGAN, and constrained_patchgan can be selected in the .json config file.
Please refer to the samples in the "sample_configs" folder.


- Bug for discriminator fixed. Now the input and target are not concatenated in the discriminator.
