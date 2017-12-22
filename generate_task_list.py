#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import exoarch

fmt = "cd /mnt/home/dforeman/research/projects/rotate ; "
fmt += "/mnt/home/dforeman/research/projects/rotate/run.sh {0}\n"

if "douglas" in sys.argv:
    with open("apjaa6e52t3_mrt.txt", "r") as f:
        with open("tasklist_praesepe.txt", "w") as fout:
            lines = f.readlines()
            for l in lines[66:]:
                fout.write(fmt.format(l[8:17]))
else:
    candidates = exoarch.K2CandidatesCatalog().df
    with open("tasklist_candidates.txt", "w") as f:
        for name in candidates.epic_name.unique():
            f.write(fmt.format(name[5:]))
