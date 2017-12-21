#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import exoarch

fmt = "cd /mnt/home/dforeman/research/projects/rotate ; "
fmt += "/mnt/home/dforeman/research/projects/rotate/run.sh {0}\n"

candidates = exoarch.K2CandidatesCatalog().df
with open("tasklist_candidates.txt", "w") as f:
    for name in candidates.epic_name.unique():
        f.write(fmt.format(name[5:]))
