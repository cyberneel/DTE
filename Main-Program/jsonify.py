# import dependencies
import torch
import requests
from transformers import pipeline

###############################################\
# NOTE: These arwe just my attempts using free
# options I found on the internet. You can
# most likely host your own local API for a
# secure option if you need the privscy.
###############################################

# this is the model we will use, better ones prob exist but need finding
generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

def GetSpecsJson(raw=None):
    # return if blank
    if raw == None:
        return
    
    # run the NLP stuff
    res = generate_text("Here is the raw text, turn this into a json with all the appropriate fields & fix any typos: " + raw, max_length=1000)
    print(res[0]["generated_text"]) 

    return res[0]["generated_text"]

# using cloudflare API for speed and for better models (Free at the time of coding)
def GetSpecJsonCloud(raw=None):
    # return if blank
    if raw == None:
        return
    
