import yaml

class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg
        
    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v
    
    def __str__(self):
        return str(self._cfg)
    
def config_from_yaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotConfig(config)   
    return config