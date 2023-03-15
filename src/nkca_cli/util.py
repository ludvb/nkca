import os
import subprocess
import sys
from contextlib import contextmanager

from nkca import __version__, session


@contextmanager
def cd(path):
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def get_version():
    try:
        with cd(os.path.dirname(__file__)):
            git_describe = subprocess.run(
                ["git", "describe", "--always", "--long", "--dirty"],
                capture_output=True,
            )
        if git_describe.returncode != 0:
            raise
        return git_describe.stdout.decode().strip()
    except:
        return __version__


def list_settings():
    settings_tbl = [("version", get_version()), ("invocation", " ".join(sys.argv))]
    settings_tbl = settings_tbl + [
        (k, str(v) if len(str(v)) < 100 else str(v)[:100] + "...")
        for k, v in session.checkpoint().items()
    ]
    return settings_tbl


def batch_iterator(iterable, batch_size):
    """Iterate over an iterable in batches of a given size."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
