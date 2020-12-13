# GAN_with_ASD(Anti-Spoofing-Detection)
Machine Learning Course Project at 2020 Fall

Project report is in the branch list, named Project report.pdf.

Group member: Ningwei Xu(nx334 N10470557) Yipei Wang(yw4576 N14021400)

Some codes were borrowed from: https://github.com/zion-king/Photo-Upsampling-via-Latent-Space-Exploration-of-Generative-Models#table-of-contents  
And: https://github.com/zeusees/HyperFAS  

### Prereqs
OpenCV 3.4.3+  
Python 3.6  
Tensorflow and Keras  
CUDA and appropriate GPU  

The environment running PULSE is needed for running this code, instructions below were provided by the original github page of PULSE:  
For the full set of required Python packages, create a Conda environment from the provided YAML, e.g.
```
conda create -f pulse.yml 
```
or (Anaconda on Windows):
```
conda env create -n pulse -f pulse.yml
conda activate pulse
```
In some environments (e.g. on Windows), you may have to edit the pulse.yml to remove the version specific hash on each dependency and remove any dependency that still throws an error after running ```conda env create...``` (such as readline)
```
dependencies
  - blas=1.0=mkl
  ...
```
to
```
dependencies
  - blas=1.0
 ...
```
Finally, you will need an internet connection the first time you run the code as it will automatically download the relevant pretrained model from Google Drive (if it has already been downloaded, it will use the local copy). In the event that the public Google Drive is out of capacity, add the files to your own Google Drive instead; get the share URL and replace the ID in the https://drive.google.com/uc?=ID links in ```align_face.py``` and ```PULSE.py``` with the new file ids from the share URL given by your own Drive file.


### The usage of the code is just the same with the PULSE, instructions below were copied from the original github page of PULSE (https://github.com/zion-king/Photo-Upsampling-via-Latent-Space-Exploration-of-Generative-Models#table-of-contents)
## Data
By default, input data for `run.py` should be placed in `./input/` (though this can be modified). However, this assumes faces have already been aligned and downscaled. If you have data that is not already in this form, place it in `realpics` and run `align_face.py` which will automatically do this for you. (Again, all directories can be changed by command line arguments if more convenient.) You will at this stage pic a downscaling factor. 

Note that if your data begins at a low resolution already, downscaling it further will retain very little information. In this case, you may wish to bicubically upsample (usually, to 1024x1024) and allow `align_face.py` to downscale for you.  

## Applying the algorithm
Once your data is appropriately formatted, all you need to do is
```
python run.py
```
Enjoy!
