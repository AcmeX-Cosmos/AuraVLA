"""
Config Loader

Utility for loading and managing YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Load and manage configuration files"""

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file

        Args:
            file_path: Path to YAML file

        Returns:
            Configuration dictionary
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        config = ConfigLoader._substitute_env_vars(config)

        return config

    @staticmethod
    def _substitute_env_vars(obj: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(obj, dict):
            return {k: ConfigLoader._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigLoader._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace ${VAR} with environment variable
            if obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.environ.get(var_name, obj)
            return obj
        else:
            return obj
