from fastapi import FastAPI
import requests
import PIL
from PIL import Image
import botnoi.resnet50 as rn
import pickle

app = FastAPI()

# output function
modFile = 'mymod.mod'
mod = pickle.load(open(modFile,'rb'))

@app.get("/img")
async def imgurl(p_image_url: str):
    #file#Image.open(image.file)
    r = requests.get(p_image_url, allow_redirects=True)
    with open("file2.png",'wb') as f:
        f.write(r.content)
    feat = rn.extract_feature("file2.png")
    res = mod.predict([feat])[0]
    return {"class":str(res)}

if __name__ == '__main__':
   import uvicorn
   uvicorn.run(app, host="3.7.8", port=8080, debug=True) 
