from django.shortcuts import render
from transformers import AutoModelForCausalLM,AutoTokenizer
from django.views.decorators.csrf import csrf_exempt
import torch,json
from django.http import JsonResponse

# Create your views here.
def index(request):
    return render(request,template_name='index.html')


tokenizer=AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model=AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# processors and gpu

# setup device for model (use GPU if available)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=model.to(device)

def chatbot_response(request):
    if request.method=='POST':
        data=json.loads(request.body)
        user_message=data.get('message')

        # tokenize user input and generate output response
        inputs=tokenizer.encode(user_message+tokenizer.eos_token,return_tensors='pt')
        attention_mask=torch.ones(inputs.shape,device=device)
        outputs=model.generate(inputs,attention_mask=attention_mask,max_length=1000,pad_token_id=tokenizer.eos_token_id)
        
        response=tokenizer.decode(outputs[:,inputs.shapes[-1]:][0],skip_special_tokens=True)
        
        return JsonResponse({'response':response})