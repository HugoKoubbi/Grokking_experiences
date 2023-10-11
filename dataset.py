import torch
import json
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ generate input and output data #####################
p=103
fraction=1.0
GEN_DATA=True
if GEN_DATA:
    equals_token = p
    x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x = x.flatten()
    y = y.flatten()
        # plus = torch.ones(x.shape, dtype=torch.int64) * plus_token
    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x, y, equals], dim=1).to(device)
    answers = ((x + y) % p).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    train, test = torch.utils.data.random_split(data, 
                                    [int(fraction * len(data)),
                                    len(data) - int(fraction * len(data))
                                    ])

    Train=[]
    for inputs, labels in train:
        inputs=inputs.cpu()
        inputs=inputs.numpy()
        a,b,_=inputs
        labels=labels.cpu()
        labels=labels.numpy()
        Train.append({'input': str(a)+' + '+str(b)+' modulo '+str(p)+'?' , 'output': str(labels)})
    Test=[]
    for inputs, labels in test:
        inputs=inputs.cpu()
        inputs=inputs.numpy()
        a,b,_=inputs
        labels=labels.cpu()
        labels=labels.numpy()
        Test.append({'input': str(a)+' + '+str(b)+' modulo '+str(p)+'?' , 'output': str(labels)})




######################## load data  into a json file  ######################

data_add = Train

with open("dataset.json", "w") as f:
     json.dump(data_add, f, indent=4)



######################## load data  into a json file  ######################
### Add natural language instruction to the generated arithmetic data using templates   
template_name = "templates/mod_add.json"
dataset_name = "dataset.json"

with open(template_name) as fp:
    template = json.load(fp)

with open(dataset_name,"rb") as test_file:
    data_original = json.load(test_file)

data_converted = []

for instance in data_original:
    
    arithmetic = instance["input"]
    
    output_dict = {}
        
    arithmetic = "the sum of " + arithmetic.replace("+", "and") 

    n=random.randint(1,500)

    instruction = template[str(1)].format(
        input = arithmetic
    )
    
    output_dict["instruction"] = instruction
    output_dict["input"] = instance["input"]
    output_dict["output"] = instance["output"]
    #output_dict["answer"] = instance["answer"]
    
    data_converted.append(output_dict)

print("Total:", len(data_converted))

with open("dataset.json", "w") as f:
    json.dump(data_converted, f, indent=4)

print("Instructions added!")
