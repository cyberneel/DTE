# import dependencies
import torch
import requests
import CONSTANTS
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
def GetSpecsJsonCloud(raw=None):
    # return if blank
    if raw == None:
        return
    
    ACCOUNT_ID = CONSTANTS.CLOUDFLARE_ACCOUNT_ID
    AUTH_TOKEN = CONSTANTS.CLOUDFLARE_AUTH_KEY

    prompt = raw

    # Get the response from cloudflare
    response = requests.post(
    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@hf/thebloke/llama-2-13b-chat-awq",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
        "messages": [
            {"role": "system", "content": "You are an assistant that reponds with a json version of the input text and nothing else! but fix any typos like when there is a O in 2019 instead of the number 0"},
            {"role": "user", "content": prompt}
        ]
        }
    )
    result = response.json()
    print(result['result']['response'])
    print(type(result))