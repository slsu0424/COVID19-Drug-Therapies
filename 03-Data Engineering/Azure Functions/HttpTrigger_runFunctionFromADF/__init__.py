from unittest import skip
import azure.functions as func
import json
import logging
import io
#import os
import zipfile
#import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


def main(req: func.HttpRequest, obj: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # get file name from HTTP request
    req_body = req.get_json()
    inputFileName = req_body.get('fileName')

    logging.info(f"Request received to extract file: {inputFileName}")

    # connect to storage account
    connection_string = "DefaultEndpointsProtocol=https;AccountName=asastgssuaefdbhdg2dbc4;AccountKey=/wDoU7XVXzCZbcVE8XjYXVfwVWVtKkUK38nFG9MhP2Vg4JEVX1j7Wws43b9NnzctMf2nGFtx8dhw+AStw3YRPA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)

    # set input blob
    blob_client = blob_service_client.get_blob_client(container="raw/FAERS", blob = inputFileName)

    # set output container
    container_client = blob_service_client.get_container_client("curated/FAERS_output_txt_AZFunction")

    # download input blob as a stream
    with io.BytesIO() as b:
        try:
            download_stream = blob_client.download_blob()
            download_stream.readinto(b)
        except ResourceNotFoundError:
            logging.info(f"No blob found")

    # open stream in READ mode
        with zipfile.ZipFile(b) as zip:
        
            # print all contents of the zip file
            logging.info(f"List of contents: {zip.namelist()}")
            
            # extract to output container
            logging.info(f"Extracting files now...")

            for filename in zip.namelist():

                logging.info(f"Extracting files now...")
                logging.info(f"Extracting file: {filename}")

                if filename.endswith('.TXT'):

                    with zip.open(filename, mode='r') as f:
                    
                        # name of file to be landed in the target container
                        if '/' in filename:
                            filename_target = filename.split('/', 1)[1]
                        else:
                            filename_target = filename
                        
                        container_client.get_blob_client(filename_target).upload_blob(f)
                else:
                    # do not process any non .txt/TXT files 
                    skip            

            logging.info(f"Done!")

    return func.HttpResponse(
            json.dumps({
                'method': req.method,
                #'url': req.URL,
                'headers': dict(req.headers),
                'params': dict(req.params),
                'get_body': req.get_body().decode()
            })
    )
