import os
import json

def dump_config(folder):
    config = {k:v for k,v in globals().items() if k.isupper()}
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)