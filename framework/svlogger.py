
import os
import platform
import warnings
import framework.wrappers

import pandas as pd

from openpyxl import load_workbook
from openpyxl.drawing.image import Image

from pathlib import Path
from datetime import datetime
from traceback import format_exc

class Logger:
    def __init__(self, path_folder, filename=None, sl = "\\", delimiter=","):
        self.path_folder = path_folder
        if filename is not None:
            self.filename = filename 
        else: 
            self.filename = "%s-%s.log" % (Path(__file__).stem, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        self.delimiter = delimiter
        self.sl = sl
 
    def log_data(self, data:[[]], folder="datalogs"):
        if not os.path.exists(f"{self.path_folder}{self.sl}{folder}"):
            os.makedirs(f"{self.path_folder}{self.sl}{folder}")
        try:
            with open(f"{self.path_folder}{self.sl}{folder}{self.sl}{self.filename}", "a", encoding="utf-8") as log:
                for row in data:
                    text = ((((('"%s"'+"%s" % self.delimiter)*len(row))[:-len(self.delimiter) if len(self.delimiter) else None]) % (tuple(row)))+"\n")
                    log.write(text.encode("utf-8", errors="ignore").decode("utf-8"))
            return(1)
        except Exception as e:
            print(format_exc())
            warnings.warn("\n[WARNING] Couldn't create or write in logfile %s" % self.filename)
            return (-1)
    
    @framework.wrappers.ftime
    def convert_to_xlsx(self, output_name:str, path_output:str, path_sheet:[(str, str)]):
        try:
            writer = pd.ExcelWriter(f"{path_output}{self.sl}{output_name}.xlsx")
            for path, sheet in path_sheet:
                df = pd.read_csv(path, encoding="utf-8")
                df.to_excel(writer, sheet_name = sheet, index=False)
            writer.close()
        except:
            print("[ERROR] Couldn't convert .scd to .xlsx", format_exc())
