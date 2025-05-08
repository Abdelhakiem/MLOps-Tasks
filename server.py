
# from src.inference.app import InferenceAPI
# import litserve as ls

# if __name__ == "__main__":
#     api = InferenceAPI()
#     server = ls.LitServer(
#         api, 
#         accelerator="cpu"
#     )
#     server.run(port=8000, generate_client_file = False)

# src/server.py
#!/usr/bin/env python3

# src/server.py
#!/usr/bin/env python3
import sys
from src.inference.app import InferenceAPI
import litserve as ls
import os
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python server.py <RUN_ID>")
    run_id = sys.argv[1]
    os.environ["RUN_ID"] = run_id

    api = InferenceAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000, generate_client_file=False)
