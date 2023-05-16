import logging
import os 
from datetime import datetime

LOG_File_Name=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
logs_path=os.path.join(os.getcwd(),"Logs",LOG_File_Name)
os.makedirs(logs_path,exist_ok=True)

log_file_path=os.path.join(logs_path,LOG_File_Name)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

