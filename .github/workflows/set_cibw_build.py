#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import os
import sys


# pylint: disable-next=consider-using-f-string
CIBW_BUILD = 'CIBW_BUILD=*cp%d%d-*manylinux*' % sys.version_info[:2]

print(CIBW_BUILD)
with open(os.getenv('GITHUB_ENV'), mode='a', encoding='utf-8') as file:
    print(CIBW_BUILD, file=file)
