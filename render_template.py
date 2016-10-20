#!/usr/bin/env python

import jinja2
import sys

print(jinja2.Template(sys.stdin.read()).render())
