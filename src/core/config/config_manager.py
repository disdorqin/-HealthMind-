"""Configuration manager for the power forecasting system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, fields
import yaml
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 3306
    username: str = "root"
    password: str = ""
    database: str = "power_trading"
    dialect: str = "mysql"
    driver: str = "pymysql"


@dataclass
class ModelConfig:
    """Model configuration."""
    input_dim: int = 19
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lookback: int = 24
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_path: str = "data/data.csv"
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "actual_power"


@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig
    model: ModelConfig
    data: DataConfig
    api_host: str = "localhost"
    api_port: int = 5000
    debug: bool = True


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        load_dotenv()
        self._config_path = config_path
        self._config: Optional[AppConfig] = None
        
    def load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        if self._config is not None:
            return self._config
            
        # Load from YAML file if exists
        yaml_config = {}
        if self._config_path and Path(self._config_path).exists():
            with open(self._config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        
        # Override with environment variables
        env_config = self._load_from_env()
        
        # Merge configurations
        merged_config = self._deep_merge(yaml_config, env_config)
        
        # Create config objects
        database_config = DatabaseConfig(
            host=merged_config.get('database', {}).get('host', 'localhost'),
            port=merged_config.get('database', {}).get('port', 3306),
            username=os.getenv('DB_USER', merged_config.get('database', {}).get('username', 'root')),
            password=os.getenv('DB_PASSWORD', merged_config.get('database', {}).get('password', '')),
            database=os.getenv('DB_NAME', merged_config.get('database', {}).get('database', 'power_trading')),
            dialect=merged_config.get('database', {}).get('dialect', 'mysql'),
            driver=merged_config.get('database', {}).get('driver', 'pymysql'),
        )
        
        model_config = ModelConfig(
            input_dim=merged_config.get('model', {}).get('input_dim', 19),
            hidden_dim=merged_config.get('model', {}).get('hidden_dim', 64),
            num_layers=merged_config.get('model', {}).get('num_layers', 2),
            dropout=merged_config.get('model', {}).get('dropout', 0.2),
            lookback=merged_config.get('model', {}).get('lookback', 24),
            epochs=merged_config.get('model', {}).get('epochs', 50),
            batch_size=merged_config.get('model', {}).get('batch_size', 64),
            learning_rate=merged_config.get('model', {}).get('learning_rate', 1e-3),
        )
        
        data_config = DataConfig(
            data_path=merged_config.get('data', {}).get('data_path', 'data/data.csv'),
            test_size=merged_config.get('data', {}).get('test_size', 0.2),
            random_state=merged_config.get('data', {}).get('random_state', 42),
            target_column=merged_config.get('data', {}).get('target_column', 'actual_power'),
        )
        
        self._config = AppConfig(
            database=database_config,
            model=model_config,
            data=data_config,
            api_host=os.getenv('API_HOST', merged_config.get('api_host', 'localhost')),
            api_port=int(os.getenv('API_PORT', merged_config.get('api_port', 5000))),
            debug=os.getenv('DEBUG', str(merged_config.get('debug', True))).lower() == 'true',
        )
        
        return self._config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Database config from env
        if os.getenv('DB_HOST'):
            config.setdefault('database', {})['host'] = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config.setdefault('database', {})['port'] = int(os.getenv('DB_PORT'))
        if os.getenv('DB_USER'):
            config.setdefault('database', {})['username'] = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            config.setdefault('database', {})['password'] = os.getenv('DB_PASSWORD')
        if os.getenv('DB_NAME'):
            config.setdefault('database', {})['database'] = os.getenv('DB_NAME')
        
        # API config from env
        if os.getenv('API_HOST'):
            config['api_host'] = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            config['api_port'] = int(os.getenv('API_PORT'))
        
        # Debug from env
        if os.getenv('DEBUG'):
            config['debug'] = os.getenv('DEBUG', '').lower() == 'true'
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @property
    def config(self) -> AppConfig:
        """Get loaded configuration."""
        if self._config is None:
            return self.load_config()
        return self._config


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get global configuration instance."""
    return _config_manager.config