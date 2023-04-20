#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import pathlib
import re


ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

VERSION_FILE = ROOT / 'torchopt' / 'version.py'

VERSION_CONTENT = VERSION_FILE.read_text(encoding='utf-8')

VERSION_FILE.write_text(
    data=re.sub(
        r'__release__\s*=.*',
        '__release__ = True',
        string=VERSION_CONTENT,
    ),
    encoding='utf-8',
)
