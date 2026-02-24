import time

from datetime import timedelta
from pathlib import Path

import hydra
import logging
import os

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from src.utils.print_utils import cyan
from src.utils.omegaconf import register_resolvers
from src.utils.logging import FileLoggingContext
console_logger = logging.getLogger(__name__)

def run_local(cfg: DictConfig):

    start_time = time.time()
    print(f"Starting application at {start_time}")
    
    # Resolve the config (project resolvers: not, equal, ifelse).
    register_resolvers()
    OmegaConf.resolve(cfg)

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    # Apply Hydra config-group choices for every group defined in configurations/
    with open_dict(cfg):
        for group_key, choice in cfg_choice.items():
            if choice is not None and group_key in cfg:
                cfg[group_key]._name = choice
    
    
    # Set up the output directory (Hydra output dir + run name from +name=...).
    output_dir = Path(hydra_cfg.runtime.output_dir) 
    with open_dict(cfg):
        if "application" in cfg:
            cfg.application.output_dir = output_dir

    # Set up experiment-level logging to file while preserving stdout.
    experiment_log_path = output_dir / "experiment.log"
    experiment_log_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLoggingContext(log_file_path=experiment_log_path, suppress_stdout=False):
        console_logger.info(f"Outputs will be saved to: {output_dir}")
        print(cyan(f"Outputs will be saved to:"), output_dir)
        
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

        # Log and save resolved configuration.
        resolved_config_yaml = OmegaConf.to_yaml(cfg)
        console_logger.info("Resolved configuration:\n" + resolved_config_yaml)
        print(cyan("Resolved configuration:"))
        print(resolved_config_yaml)

        # Save config to output directory for reproducibility.
        config_file = output_dir / "resolved_config.yaml"
        with open(config_file, "w") as f:
            f.write(resolved_config_yaml)
        console_logger.info(f"Saved resolved config to: {config_file}")
        print(cyan(f"Saved resolved config to: {config_file}"))

        # Launch experiment.
        console_logger.info("Starting application execution")


        console_logger.info(
            "Experiment execution completed in "
            f"{timedelta(seconds=time.time() - start_time)}"
        )
    
@hydra.main(version_base=None, config_path="configurations", config_name="config")
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError(
            "Must specify a name for the run with command line argument '+name=[name]'"
        )

    # Configure logging level from LOGLEVEL environment variable.
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console_logger.info(f"Logging level: {log_level}")
    run_local(cfg)


if __name__ == "__main__":
    run()