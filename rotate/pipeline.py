# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["Pipeline"]


class Pipeline(object):

    def __init__(self):
        self._children = []

    def add_child(self, child):
        self._children.append(child)

    def run(self, inputs):
        result = self.process(inputs)
        if len(self._children) == 0:
            return result
        elif len(self._children) == 1:
            return self._children[0].run(result)
        return [child.run(result) for child in self._children]

    def process(self, inputs):
        raise NotImplementedError("subclasses must implement this method")
