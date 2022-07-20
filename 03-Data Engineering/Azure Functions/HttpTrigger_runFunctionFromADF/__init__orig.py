import logging

import azure.functions as func

import pandas as pd


def main(req: func.HttpRequest, obj: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('fileName')

    logging.info(f"File name is: {name}")

    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('fileName')
    
    logging.info(f"File name is: {name}")

    #logging.info(f'Python HTTP triggered function processed: {obj.read()}')

    # df = pd.read_csv({name})
    # df.to_parquet('output.parquet')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )


    #logging.info(f"Python blob trigger function processed blob \n"
    #             f"Name: {myblob.name}\n"
    #             f"Blob Size: {myblob.length} bytes")
    #logging.info('Python Blob trigger function processed %s', myblob.name)