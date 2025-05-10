import sys
from src.inference.app import InferenceAPI
import litserve as ls

if __name__ == "__main__":
    api = InferenceAPI()
    server = ls.LitServer(
        api, 
        accelerator="cpu"
    )
    server.run(port=8080, generate_client_file = False)    

# {
#   "input": {
#     "PassengerId": 1,
#     "Survived": 0,
#     "Pclass": 3,
#     "Name": "Braund, Mr. Owen Harris",
#     "Sex": "male",
#     "Age": 22.0,
#     "SibSp": 1,
#     "Parch": 0,
#     "Ticket": "A/5 21171",
#     "Fare": 7.25,
#     "Cabin": null,
#     "Embarked": "S"
#   }
# }
