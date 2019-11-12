#!/usr/bin/env python

# convert nested dictionary to python object
class mapper:
    def __init__(self, **response):
        for k, v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = mapper(**v)
            else:
                self.__dict__[k] = v