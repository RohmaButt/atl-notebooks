pip install streamlit streamlit-webrtc vitallens opencv-python av
pip install scipy
pip install psutil
python examples/live.py --method=VITALLENS --api_key=GdsxHv55ys3pWHUEc8tZm5Sr6H5oeb7n30NouKyX

import os
os.environ["VITALLENS_API_KEY"] = "GdsxHv55ys3pWHUEc8tZm5Sr6H5oeb7n30NouKyX"
os.environ["NGROK_AUTHTOKEN"]= "36wHEXlreXD3xW4W6KDRapNoy30_PVQCqzWTztVEwrP6CB88"

Install this in env with python version >=3.9

