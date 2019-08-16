import logging

import shutil

import os

import tempfile


class TempFilename(object):
    def __init__(self):
        """
        Create a temp filename. The file is guaranteed to exist in FS.
        """
        fd, self.filename = tempfile.mkstemp()
        os.close(fd)
        logging.debug('Created temp file: %s', self.filename)

    def __enter__(self):
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.unlink(self.filename)
        logging.debug('Removed temp file: %s', self.filename)


class TempDirname(object):
    def __init__(self):
        """
        Create a temp dir. The dir is guaranteed to exist in FS.
        """
        self.dirname = tempfile.mkdtemp()
        logging.debug('Created temp dir: %s', self.dirname)

    def __enter__(self):
        return self.dirname

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.dirname)
        logging.debug('Removed temp dir: %s', self.dirname)
