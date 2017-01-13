'''
Set the global config variable.
'''

import configparser
import json
import re


class Config(dict):
    _parser = configparser.ConfigParser()
    _filename = ""
    _data = {}

    def __init__(self, *args, **kwargs):
        _filename = kwargs.get("filename") or args[0]

        if _filename:
            self.read(_filename)

    def __getitem__(self, key):
        return self._data[key]

    def __parse__(self, _dict, _container):
        for key, value in _dict.items():
            if type(value) is str:
                _container[key] = self.__type__(value)
            elif type(value) is configparser.SectionProxy:
                _container[key] = {}
                self.__parse__(_dict[key], _container[key])
            else:
                _container[key] = value

    def __type__(self, value):
        if re.search("^\[.*\]$", value) is not None:
            value = json.loads(value)
        elif re.search("^\{.*\}$", value) is not None:
            value = json.loads(value)
        elif re.search("^\d*\.\d*$", value) is not None:
            value = float(value)
        elif re.search("^\d+$", value) is not None:
            value = int(value)
        elif value == "True" or value == "False":
            value = True if value == "True" else False
        return value


    def read(self, filename):
        self._parser.read(filename)
        self.__parse__(self._parser, self._data)

