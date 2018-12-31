"""Utilities for using sacred experiments with FileStorageObservers"""

import os
import json
import importlib.util as iutil

# %%

def remove_key(d, key):
    """Remove the key from the given dictionary and all its sub dictionaries.
    Mutates the dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.
    key : type
        Key to recursively remove.

    Returns
    -------
    None
    """
    for k in list(d.keys()):
        if isinstance(d[k], dict):
            remove_key(d[k], key)
        elif k == key:
            del d[k]

def read_config(run_dir):
    """Read the config.json from the given run directory. Removes __doc__
    entries and returns a config dict.

    Parameters
    ----------
    run_dir : str
        Path to run directory.

    Returns
    -------
    dict
        Config dict with __doc__ entries removed.

    """
    with open(os.path.join(run_dir, 'config.json')) as file:
        config = json.loads(file.read())
        remove_key(config, key='__doc__')

    return config

def get_model_path(run_dir, epoch):
    """Get the path to the saved model state_dict with the given epoch number.
    If epoch is 'latest', the latest model state dict path will be returned.

    Parameters
    ----------
    run_dir : str
        Path to run directory.
    epoch : type
        Epoch number of model to get.

    Returns
    -------
    str
        Path to model state_dict file.
    """
    if epoch == 'latest':
        return os.path.join(run_dir, 'latest.statedict.pkl')

    filenames = os.listdir(run_dir)

    for filename in filenames:
        if 'statedict' not in filename:
            continue
        if filename.startswith('epoch'):
            number = int(filename[len('epoch'):].split('_')[0])
            if epoch == number:
                return os.path.join(run_dir, filename)

    raise ValueError(f"No statedict found with epoch number '{epoch}'")

# %%

def import_source(run_dir, module_name):
    """Import a module used in a sacred run, from the "_sources" directory.

    Parameters
    ----------
    run_dir : str
        Path to directory of the sacred run.
    module_name : str
        Name of the module whose source to import (without .py).

    Returns
    -------
    module
        Module which was used in the run.

    """
    run_json_file = os.path.join(run_dir, "run.json")
    with open(run_json_file) as file:
        run = json.loads(file.read())

    sources = run['experiment']['sources']
    for source in sources:
        if source[0] == module_name+".py":
            source_file = os.path.split(source[1])[-1]
            break

    path_to_source = os.path.join(os.path.split(run_dir)[0], "_sources", source_file)

    spec = iutil.spec_from_file_location(module_name,
                                         location=path_to_source)
    module = iutil.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
