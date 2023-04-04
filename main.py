from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin
import Utilities
from Utilities import DownloadData
from Utilities import UploadWeightsForUser


cred = credentials.Certificate("vertopsecretcertificate.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

dataReturn = DownloadData(db, downloadToComputer=False, vanillaLibSvm=False)

#Upload ML weights to database.
UploadWeightsForUser(db, None, dataReturn)

#Test Data per user.
#TestData(dataReturn, test_size=.3, eliminate3=True, moodThreshold=3, show_DT_plot=False)





    


    

    

