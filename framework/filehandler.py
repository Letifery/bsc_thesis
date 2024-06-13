import framework.wrappers
import configparser
import warnings
import platform
import sys 
import re
import json

from datetime import datetime
from typing import List
from traceback import format_exc
from framework.svlogger import Logger
from pathlib import Path

class DataLoader():
    def __init__(self, path_folder, data_path = "config.ini", sl = "\\"):
        self.cfg_parser = configparser.ConfigParser()
        self.error_logger = Logger(path_folder=path_folder, filename="errorlogs", sl=sl)
        self.sl = sl
        self.data_path = data_path
        if self.cfg_parser.read(self.data_path) == [] and self.data_path=="config.ini":
            self.generate_default_config()
            self.cfg_parser.read(self.data_path)
            
    @framework.wrappers.ftime    
    def generate_default_config(self):
        try:
            self.cfg_parser["general"] = {"path_input":"input", 
                                          "path_output":"output", 
                                          "path_images_output":"output_images",
                                          "path_logs" :"logs",
                                          "name_output":"scd_analysis_v0.9.4",
                                         "docx2pdf_flag":False, 
                                         "reset_data_at_start":True,
                                         "limit_files_to_analyse":"None",
                                         "shuffle_files":False,
                                         "skip_ui_on_startup":False,
                                         "name_subcost_xlsx_file": "all_subcosts"}
                                         
            self.cfg_parser["language detection"] = {"mode":"random", 
                                                    "num_paragraphs":50, 
                                                    "threshold":0.5, 
                                                    "skip_length":25}
            with open(self.data_path, "w") as cfg:
                self.cfg_parser.write(cfg)
        except Exception as e:
            warnings.warn("\n[ERROR] Couldn't generate config")
            self.error_logger.filename = "gen-error-cfg-%s.log" % (datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
            self.error_logger.log_data([[format_exc()]], "errorlogs")
            sys.exit()
         
         
    @framework.wrappers.ftime
    def load_config(self, file:str = ""):
        try:
            self.cfg_parser.read(self.data_path+file) 
            self.cfg = {s:dict(self.cfg_parser.items(s)) for s in self.cfg_parser.sections()}
            
        except KeyError:
            warnings.warn("\n[ERROR] Couldn't load config")
            self.error_logger.filename = "load-error-cfg-%s.log" % (datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
            self.error_logger.log_data([[format_exc()]], "errorlogs")
            sys.exit()
    
    def set_section(self, section_name:str, keys:List[str], values:List[str]):
        """Can change one or more values in a specific section (section name) in the config.ini file"""
        for key, value in zip(keys, values):
            self.cfg_parser.set(section_name, key, value)

        with open('config.ini', 'w') as configfile:
            self.cfg_parser.write(configfile)
    
    def get_section(self, section_name:str, keys:[str]):
        """Get one or more values in a specific section (section name) in the config.ini file"""
        section_values = []
        for key in keys:
            section_values.append(self.cfg_parser.get(section_name, key))
        return section_values

    @framework.wrappers.ftime
    def get_section_general(self):
        try:
            presets_general = [self.cfg_parser.get('general', 'path_input').split(','),
                               self.cfg_parser.get('general', 'path_output'),
                               self.cfg_parser.get('general', 'path_logs'),
                               self.cfg_parser.get('general', 'name_output'),
                               self.cfg_parser.getboolean('general', 'docx2pdf_flag'),
                               self.cfg_parser.getboolean('general', 'reset_data_at_start'),
                               self.cfg_parser.getboolean('general', 'shuffle_files'),
                              ]
            return(presets_general)
            
        except KeyError:
            warnings.warn("\n[ERROR] Couldn't read config")
            self.error_logger.filename = "read-error-cfg-%s.log" % (datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
            self.error_logger.log_data([[format_exc()]], "errorlogs")
            sys.exit()
    
    
    def get_json_data(self, path2json_file:str, section_names:List[str]) -> List[str]: 
        """Returns data from all specified sections of a json file"""
        f = open(path2json_file)
        json_data = json.load(f)
        data  = [] 
        for section_name in section_names:
            data.append(json_data[section_name])
        return data 