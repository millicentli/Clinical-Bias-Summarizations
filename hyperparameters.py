# Hyperparameters to run experiments

HYPERPARAMETERS = {
    "train": True,
    "evals": False,
    "train_data_file": "mimic-cxr/files/data.txt",
    "model_type": "facebook/bart-large-cnn",
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "learning_rate": 1e-3,
    "num_trained_epochs": 1,
    "grad_clip": 0.25,
    "output_dir": "",
    "output_file_name_data": "",
    "output_file_name_labels": ""
}
