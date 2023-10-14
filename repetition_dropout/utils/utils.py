from header import *
import torch
import json
import os
import yaml
import os.path as path

class FileType:
    PT = "pt"
    TXT = "txt"
    JSON = "json"
    TSV = "tsv"
    TAR = "tar"
    YAML = "yaml"
    CSV = "csv"
    NPY = "npy"
    ALL = ["pt", "txt", "json", "tsv", "tar", "yaml", "csv"]


class FileExtensionType:
    ADD = "add"
    CHANGE = "change"


LANGUAGE_LIST = ["en", "de", "zh", "vi", "fr"]



def generate_mask(ids, pad_token_idx=0):
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] == 0
    return mask



class FileUtils:
    @staticmethod
    def exists(file_path):
        return path.exists(file_path)
    
    @staticmethod
    def rename(source_fname, target_fname):
        if os.path.exists(source_fname):
            # Rename the file
            os.rename(source_fname, target_fname)
            logging.info("File renamed from {} to {}".format(source_fname, target_fname))
        else:
            logging("The file {} does not exist".format(source_fname))

    @staticmethod
    def is_dir(file_path):
        return path.isdir(file_path)

    @staticmethod
    def get_last_path(path):
        if not path:
            return path
        parent, last_path = os.path.split(path)
        if last_path:
            return last_path
        else:
            return FileUtils.get_last_path(parent)

    @staticmethod
    def get_dir(path):
        return os.path.dirname(path)

    @staticmethod
    def check_dirs(dir_path):
        if path.exists(dir_path):
            logging.info("{} already exists".format(dir_path))
        else:
            logging.info("Making new directory {}".format(dir_path))
            os.makedirs(dir_path)

    @staticmethod
    def check_basename(fpath):
        bname = os.path.basename(fpath)
        parts = bname.split(".")
        if len(parts) <= 1:
            return bname
        elif parts[-1] in LANGUAGE_LIST or parts[-1] in FileType.ALL:
            return ".".join(parts[:-1])
        else:
            return bname

    @staticmethod
    def check_file_type(fpath):
        parts = fpath.split(".")
        ext = ""
        if parts:
            ext = parts[-1]
        return ext