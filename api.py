from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import os

app = FastAPI()
torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        max_length = json_post_raw.get('max_length')
        top_p = json_post_raw.get('top_p')
        temperature = json_post_raw.get('temperature')
        messages = json_post_raw.get('messages')

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_new_tokens=1500, do_sample=True)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        pos=response.find(messages[0]['content'])
        if pos!=-1:
            response=response[pos+len(messages[0]['content']):]
        else:
            response=response
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                }
            }],
        }
        log = f"[{time}] prompt: {messages[0]['content']}, response: {repr(response)}"
        print(log)
        return answer

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

if __name__ == '__main__':

    model_dir="Qwen/Qwen2.5-7B-Chat",
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
