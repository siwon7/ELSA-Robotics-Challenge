"""ELSA/FLAME Ralph verification script.

Validates code changes without GPU training:
1. Python syntax check on modified agent code
2. YAML config validation (all experiment configs)
3. Model instantiation smoke test (CPU, no data)
"""

import subprocess
import sys
import pathlib
import json

ROOT = pathlib.Path(__file__).parent


def run(cmd, cwd=None, check=True):
    print(f"> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(result.returncode)
    return result


def check_syntax():
    """Layer 1: Python syntax check on core files."""
    core_files = [
        "elsa_learning_agent/agent.py",
        "elsa_learning_agent/config_utils.py",
        "elsa_learning_agent/config_validation.py",
        "elsa_learning_agent/dataset/dataset_loader.py",
        "scripts/train_same_env_bcpolicy_probe.py",
    ]
    for f in core_files:
        path = ROOT / f
        if path.exists():
            run([sys.executable, "-m", "py_compile", str(path)])
    print("Syntax check passed.")


def check_configs():
    """Layer 2: Validate all experiment YAML configs parse correctly."""
    import yaml

    config_dir = ROOT / "experiments"
    if not config_dir.exists():
        print("No experiments/ directory found, skipping config check.")
        return

    errors = []
    for yml in sorted(config_dir.glob("*.yaml")):
        try:
            with open(yml) as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                errors.append(f"{yml.name}: not a dict")
                continue
            if "model" not in cfg or "dataset" not in cfg:
                errors.append(f"{yml.name}: missing 'model' or 'dataset' section")
        except Exception as e:
            errors.append(f"{yml.name}: {e}")

    if errors:
        for e in errors:
            print(f"  Config error: {e}")
        sys.exit(1)
    print(f"Config check passed ({len(list(config_dir.glob('*.yaml')))} configs).")


def check_model_instantiation():
    """Layer 3: Smoke test - can we create the model on CPU?"""
    try:
        sys.path.insert(0, str(ROOT))
        from elsa_learning_agent.agent import Agent

        # Test with default CNN config (lightest)
        agent = Agent(
            image_channels=3,
            low_dim_state_dim=8,
            action_dim=8,
            image_size=(64, 64),
            vision_backbone="cnn",
            policy_head_type="mlp",
        )
        print("Model instantiation (CNN+MLP) passed.")

        # Test diffusion head
        agent_diff = Agent(
            image_channels=3,
            low_dim_state_dim=8,
            action_dim=8,
            image_size=(64, 64),
            vision_backbone="cnn",
            policy_head_type="diffusion",
            diffusion_num_steps=5,
        )
        print("Model instantiation (CNN+Diffusion) passed.")

        # Test separate gripper head
        agent_split = Agent(
            image_channels=3,
            low_dim_state_dim=8,
            action_dim=8,
            image_size=(64, 64),
            vision_backbone="cnn",
            policy_head_type="diffusion",
            diffusion_num_steps=5,
            separate_gripper_head=True,
            gripper_loss_weight=4.0,
        )
        print("Model instantiation (CNN+Diffusion+SplitGripper) passed.")

    except Exception as e:
        print(f"Model instantiation failed: {e}")
        sys.exit(1)


def main():
    check_syntax()
    check_configs()
    check_model_instantiation()
    print("\nVERIFY PASSED")


if __name__ == "__main__":
    main()
