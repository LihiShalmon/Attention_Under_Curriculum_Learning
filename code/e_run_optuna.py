import optuna
import wandb
import torch
import numpy as np
import random
from d_run_full_exp import RunExp  # Ensure this import works

curriculum_types = ["vanilla", "curriculum", "anti"]
models = ["resnet20", "resnet20_selfattn" ] #"resnet20_mhattn", "resnet20_cbam"]

wandb.init(project="curriculum-learning-tuning")
def run_experiment(curriculum, model, learning_rate, weight_decay, num_epochs, start_fraction, inc, step_length, starting_portion):
    exp = RunExp(
        dataset="cifar100_subset_16",
        curriculum=curriculum,
        run_name=f"{curriculum}_{model}",
        used_model=model,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_decay_rate=0.9,
        weight_decay=weight_decay,
        start_fraction=start_fraction,
        inc=inc,
        step_length=step_length,
        starting_portion=starting_portion
    )
    return exp.run_experiment()  

def objective(trial):
    config = wandb.config  # Get hyperparameters from W&B
    learning_rate = trial.suggest_uniform("learning_rate", config.learning_rate["min"], config.learning_rate["max"])
    weight_decay = trial.suggest_uniform("weight_decay", config.weight_decay["min"], config.weight_decay["max"])
    start_fraction = trial.suggest_uniform("start_fraction", config.start_fraction["min"], config.start_fraction["max"])
    inc = trial.suggest_uniform("inc", config.inc["min"], config.inc["max"])
    step_length = trial.suggest_int("step_length", config.step_length["min"], config.step_length["max"])
    starting_portion = trial.suggest_uniform("starting_portion", config.starting_portion["min"], config.starting_portion["max"])

    for curriculum in curriculum_types:
        for model in models:
            num_epochs = 50  # Fixed number of epochs

            # Run Experiment
            test_loss = run_experiment(curriculum, model, learning_rate, weight_decay, num_epochs, start_fraction, inc, step_length, starting_portion)

            # ✅ Log results to W&B
            wandb.log({
                "curriculum": curriculum,
                "model": model,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "start_fraction": start_fraction,
                "inc": inc,
                "step_length": step_length,
                "starting_portion": starting_portion,
                "test_loss": test_loss,  # ✅ Log test loss
            })

    return test_loss  # Minimize test loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=36)  # Runs multiple trials with different hyperparameters

print("Best Hyperparameters:", study.best_params)
wandb.finish()
