import os


class DataPath:
    PREFIX = "/home/turja"
    TEMPORAL_MAPPING_FILE = os.path.join(PREFIX, "AD_files", "temporal_mapping_baselabel.json")
    PET_DATA = os.path.join(PREFIX, "AD_files", "PET_data.json")
    NETWORK_DATA = os.path.join(PREFIX, "AD_network_raw")
    AMYLOID_PATH = os.path.join(PREFIX, "Amyloid_Smooth_SUVR.xlsx")
    PHENO_DATA_PATH = os.path.join(PREFIX, "AD_files", "MRI_information.xlsx")
    PARC_TABLE = os.path.join(PREFIX, "AD_files", "parcellationTable_Ordered.json")