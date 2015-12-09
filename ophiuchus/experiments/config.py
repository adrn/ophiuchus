# coding: utf-8

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import yaml

__all__ = ['ConfigNamespace']

class ConfigNamespace(object):

    def __init__(self):
        # do nothing...
        pass

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def keys(self):
        return self.__dict__.keys()

def save(ns, f):
    """
    Either return a YAML string representing the configuration state
    or save the configuration state to the specified YAML file.

    Parameters
    ----------
    ns : ConfigNamespace
        The namespace object.
    f : str, file-like
        Either a file path (as a string) or a file-like object with a
        `write()` method.
    """
    if hasattr(f, 'write'):
        yaml.dump(dict(ns), f, default_flow_style=False)
    else:
        with open(f, 'w') as f:
            yaml.dump(dict(ns), f, default_flow_style=False)

def load(f):
    """
    Load a configuration state from a given file name or file-like
    object.

    Parameters
    ----------
    f : str, file-like
        Either a file path (as a string) or a file-like object with a
        `read()` method.

    Returns
    -------
    ns : ConfigNamespace
        Returns a ConfigNamespace object with the loaded configuration
        parameters.
    """
    if hasattr(f, 'read'):
        p_dict = yaml.load(f.read())
    else:
        with open(os.path.abspath(f)) as fil:
            p_dict = yaml.load(fil.read())

    ns = ConfigNamespace()
    for k,v in p_dict.items():
        setattr(ns, k, v)
    return ns
