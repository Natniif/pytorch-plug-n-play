# pytorch-plug-n-play
Pytorch lightning code that allows you to just set up your dataset and model and you are good to go.
I set this up so that new projects can get running and off the ground quickly and also because I hate re-writing boilerplate code.

## Quick tour
The project is configured from .yaml files found in the `config/` directory. I recommend you create a new config for each experiment that you run so that you can track past configurations (even though these are recorded on wandb anyways)

The code runs from `run.py` which can be ran as `python -m plug.run "YOUR_CONFIG.yaml"` after which the checkpoints will be created in the `checkpoints/` directory as written in `checkpoin_dir` in the yaml config file. 

`sweep.py` is a file to use for hyperperameter tuning. 

## Usage 

1. Create conda environment using `conda create -n PROJECT_NAME` and run `pip install -e .` to install all dependencies in setup.py
2. Create Pytorch model that inherits from the BaseModel class in plug.models and import into `run.py`
3. Vice versa for datasets and dataloaders but there is no class to inheret from. (I recommend you use pytorch lightnings dataloader superclass)

## Small notes
- The default dtype for tensors is set as float32 on medium accuracy in the `run.py` file which you will have to change manually within the file. 