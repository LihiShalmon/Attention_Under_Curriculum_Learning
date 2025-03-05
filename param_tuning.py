import time
import wandb
from d_run_full_exp import RunExp
from multiprocessing import freeze_support

# Define the combinations you want to tune for:
model_types = ["resnet20_selfattn", "resnet20_cbam"]
curriculum_types = ["vanilla", "curriculum", "anti-curriculum"]
used_model = "resnet20_selfattn"
curriculum = "curriculum"

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing

    fixed_epochs = 20  # Fixed number of epochs for each tuning run


    # Create a unique run name using the current time in milliseconds.
    unique_id = int(time.time() * 1000)
    run_name = f"{used_model}_{curriculum}_hptuning_{unique_id}"

    # Initialize WandB run with a unique name and group tag so that each run is separate.
    wandb.init(
        project="CNN_With_Curriculum_Learning",
        entity="lihi-shalmon-huji-hebrew-university-of-jerusalem-org",
        name=run_name,
        group=f"{used_model}_{curriculum}"
    )

    # Retrieve hyperparameters from wandb.config (these are tuned by the sweep agent)
    config = wandb.config

    # Create an experiment instance with the tuned hyperparameters.
    # (Ensure that RunExp.__init__ accepts pacing_function, start_fraction, inc, and step_length.)
    exp = RunExp(
        dataset="cifar100_subset_16",
        curriculum=curriculum,
        run_name=run_name,
        num_epochs=fixed_epochs,
        used_model=used_model,
        num_classes=5,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size if hasattr(config, "batch_size") else 128,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        lr_decay_rate=config.lr_decay_rate,
        pacing_function=config.get("pacing_function", "exponential"),
        start_fraction=config.get("start_fraction", 0.3),
        inc=config.get("inc", 1.5),
        step_length=config.get("step_length", 1)
    )

    # Run the experiment (each run logs separately to WandB).
    exp.run_experiment()
    wandb.finish()
