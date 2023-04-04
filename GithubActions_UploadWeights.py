from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin
import Utilities
from Utilities import DownloadData
from Utilities import UploadWeightsForUser

# This file was used with github actions to automate the upload of weights to the database.
# every day it would automatically download the data from firebase and process it though 
# my model. Then it would upload the weights to the database. This was used by the app to
# show users their weight predictions for the next day.


cred = credentials.Certificate("VeryTopSecretCertificate.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

dataReturn = DownloadData(db, downloadToComputer=False, vanillaLibSvm=False)


#Upload ML weights to database.
UploadWeightsForUser(db, None, dataReturn)





    


    

    

