"""Utilities for using sacred experiments with FileStorageObservers"""

import os
import json
import logging
import sys
import importlib
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location


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

class SacredFinder(MetaPathFinder):
    """MetaPathFinder to perform imports based on sources used by sacred runs.
    Loads list of sources from the run.json of the given run_dir and tries to
    import the sources used in this run. Does not work for packages currently.

    Parameters
    ----------
    run_dir : str
        Path of run directory. Assumes _sources is present in the parent directory.
    _log : Logger
        Logger instance for logging.
    """
    def __init__(self, run_dir, _log=logging.getLogger("sacred_finder")):
        run_json_file = os.path.join(run_dir, "run.json")
        with open(run_json_file) as file:
            run = json.loads(file.read())

        self.sources = run['experiment']['sources']
        self.parent_dir = os.path.join(os.path.split(run_dir)[0])
        self._log = _log

    def find_spec(self, fullname, path, target=None):
        for source in self.sources:
            if source[0] == fullname+".py":
                module_location = os.path.join(self.parent_dir, source[1])
                self._log.info(f"Imported {fullname} from {module_location}")
                return spec_from_file_location(fullname, location=module_location)

# %%

def import_source(run_dir, module_name):
    """Import a module used in a sacred run, from the "_sources" directory.
    Manages the dependencies between multiple sources of the same run using a
    MetaPathFinder. Does not work for packages (including namespace packages)
    currently.

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
    finder = SacredFinder(run_dir)
    sys.meta_path.insert(0, finder)
    module = importlib.import_module(module_name)
    del sys.meta_path[0]
    return module
