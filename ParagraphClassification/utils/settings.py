import logging
import os

import yaml


def load_yaml_as_dict(yaml_fn):
    with open(yaml_fn) as f:
        logging.info('Loading settings from %s', yaml_fn)
        info_dict = yaml.load(f)
        return info_dict


def find_settings(path=None):
    filename = '.synthesisproject.yaml'

    def try_load_file(file_path):
        if not os.path.exists(file_path):
            return None
        try:
            return load_yaml_as_dict(file_path)
        except yaml.YAMLError:
            raise RuntimeError('Cannot parse yaml file: ' + file_path)

    settings = None
    if path is not None:
        settings = try_load_file(path)

    if settings is None:
        settings = try_load_file(filename)

    if settings is None:
        settings = try_load_file(
            os.path.expanduser(os.path.join('~', filename))
        )

    if settings is None:
        raise RuntimeError('No yaml file found. '
                           'You can either put %s in your home folder, '
                           'or in your current directory' % filename)

    return settings


_cached_default_settings = None


def default_settings():
    global _cached_default_settings
    if _cached_default_settings is None:
        _cached_default_settings = find_settings()

    return _cached_default_settings

