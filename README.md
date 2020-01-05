# Echo View Converter with Clinically Valid Constraints   

Code for the paper: https://arxiv.org/pdf/1912.03120.pdf 

A Study into Echocardiography View Conversion, accepted to MedNeurIPS 2019.


To initiate training, run 

     python3 src/main.py --dataset_path=$DATASETS/CAMUS --config=configs/config_2CH_4CH.json
     
The environment variable `$DATASET` is assumed to be set to 
where the CAMUS dataset directory is stored. 


- Unified all modes of the codes.
Now segmentation, patchGAN, and constrained_patchgan can be selected in the .json config file.
Please refer to the samples in the "sample_configs" folder.


- Bug for discriminator fixed. Now the input and target are not concatenated in the discriminator.


Abstract:

Transthoracic echo is one of the most common means of cardiac studies in theclinical routines. During the echo exam, the sonographer captures a set of standardcross sections (echo views) of the heart. Each 2D echo view cuts through the 3Dcardiac geometry via a unique plane.  Consequently, different views share somelimited information.  In this work, we investigate the feasibility of generating a2D echo view using another view based on adversarial generative models.  Theobjective  optimized  to  train  the  view-conversion  model  is  based  on  the  ideasintroduced by LSGAN, PatchGAN and Conditional GAN (cGAN). The size andlength of the left ventricle in the generated target echo view is compared againstthat of the target ground-truth to assess the validity of the echo view conversion.Results show that there is a correlation of 0.50 between the LV areas and 0.49between the LV lengths of the generated target frames and the real target frames.


Sample Results:

![Sample Results](https://raw.githubusercontent.com/amir-abdi/echo-view2view/master/samples/Sample%20View%20Conversion%20Results.jpg)
