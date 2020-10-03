import argparse
import json
import collections.abc
import tempfile


def read_json(config_path):
    with open(config_path) as fp:
        config = json.load(fp)

    return config


def update_config_file(config_path, json_args):
    config = read_json(config_path)
    d = json.loads(json_args)
    new_config = update_dict(config, d)
    input_file = tempfile.NamedTemporaryFile(delete=False)
    input_file.write(json.dumps(new_config, sort_keys=True, indent=4).encode("utf-8"))
    input_file.flush()
    input_file.close()

    return input_file.name


def update_dict(d, u):
    """
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_str", type=str, default=None)
    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    new_config = update_config_file(args.config_path, args.json_str)
    print(new_config)
