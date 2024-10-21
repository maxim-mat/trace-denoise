# trace-denoise

This is the implementation for the Diffusion Denoising Trace Recovery research.

Usage:
1. Clone the repository: 
	git clone https://github.com/maxim-mat/trace-denoise.git
	cd trace-denoise/src
2. Install requirements:
	pip install -r requirements.txt
3. Run the main program:
	python main.py [options]
	
Command-Line Arguments:
- --cfg_path (optional) Path of configuration json. Default: ./config.json
- --save_path (optional) Location where outputs such as checkpoints and results will be saved. Default: summary_path from configuration json.
- --resume (optional) Boolean flag to resume a previous training run.

Configuration json:
Most of the functionality is handled through a configuration json mentioned above. By default, there exists a config.json in the root directory which can be re-configured.

Configuration Variables:
- data_path (string): Path of a compatible dataset pickle file. The file is expected to be a dictionary with keys "target" and "stochastic" for dk and sk traces with array or tensor values.
- summary_path (string): Location where logs will be written. Also default location to save results to unless specified otherwise in arguments.
- device (string): Device used for training and inference. Legal values: "cpu", "cuda", "cuda:X".
- prallelize (boolean): If set, will parallelize the model over multiple devices. In this case, set device to "cuda:X,Y,Z..."
- num_epochs (int): Number of training epochs.
- learning_rate (float): Training learning rate.
- num_timesteps (int): Number of Diffusion timesteps.
- train_percent (float): Percentage of dataset used to train the model. 0 < train_percent < 1.
- num_workers (int): Number of threads used by the data loader.
- test_every (int): Number of epochs required for evaluation. For example, for a value of 100, the model will be evaluated on the test data every 100 epochs.
- denoiser (string): Denoiser type. Use only "unet_cond" at this time.
- denoiser_hidden (int): Number of hidden layers in denoiser.
- denoiser_layers (int) [Depracated]: Number of total denoiser layers. Used only by simple MLP denoiser.
- num_classes (int): Total number of classes (activities) in dataset + 1 for padding.
- batch_size (int): Batch size for data loading.
- conditional_dropout (float): Dropout probability when training the conditional denoiser.
- eval_train (boolean): If set, a sample training batch will be evaluated along with test set in evaluation epochs.
- mode (string): Mode of generation ("cond" or "uncond"). Use "cond" for trace recovery.
- predict_on (string): Prediction target of denoiser. Can be either "original" or "noise" only. Changes the denoising behaviour, currently predicting "original" has better results.
- seed (int): Seed for dataset splitting for reproducability. Seed used in research: 42.

Additional configuration variables that are not mentioned are either for future features or unused. 
 
