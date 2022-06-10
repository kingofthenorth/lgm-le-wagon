import yaml
from dotenv import find_dotenv

env_vars = {}
with open(find_dotenv()) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.strip().split('=', 1)
        env_vars[key]=  value

with open(r'env.yaml', 'w') as file:
    documents = yaml.dump(env_vars, file)