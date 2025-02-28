from typing import Dict
from tqdm import tqdm
from microbench import MicroBench
from openai import OpenAI
from pydantic import BaseModel

from src.datasets import encode_image


class OrigScenario(BaseModel):
    origscenario: str

class OrigScenarioList(BaseModel):
    origscenarios: list[OrigScenario]
    
class Scenario(BaseModel):
    translated_scenario: str

class ScenarioList(BaseModel):
    scenarios: list[Scenario]

lpbench = MicroBench()  

class Generation:
    def __init__(self, model_type, system_prompt, user_prompt, max_tokens, api_key):
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt 
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
    
    @lpbench
    def generate_response(self, base64_image, few_shot_k, with_image) -> str:
        if few_shot_k == 0:
            messages=[
                  {"role": "system", "content": self.system_prompt},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ]
            
        else:
          if with_image:
            messages = [
                   {"role": "system", "content": self.system_prompt},
            ]
              
            for i in range(few_shot_k):
              messages.append({"role": "user", "content": self.user_prompt[i][1]})
              base64_image_fewshot = encode_image(self.user_prompt[i][0])
              messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_fewshot}"}}]})
             
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]})
              
          else:
              messages=[
               {"role": "system", "content": self.system_prompt},
               {"role": "user", "content": self.user_prompt},
               {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
              ]
              
        response = self.client.beta.chat.completions.parse(
          model=self.model_type,
          messages=messages,  
          temperature=0.0,
          max_tokens=self.max_tokens,
          response_format=OrigScenarioList
        )
                
        return response.choices[0].message.parsed

    @lpbench
    def generate_deepcontexts(self, base64_image, text_input, few_shot_k, with_image) -> str:
        if few_shot_k == 0:
            messages=[
                  {"role": "system", "content": self.system_prompt},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                                               {"type": "text", "text": text_input}]}
            ]
            
        else:
          if with_image:
            messages = [
                   {"role": "system", "content": self.system_prompt},
            ]
              
            for i in range(few_shot_k):
              base64_image_fewshot = encode_image(self.user_prompt[i][0])
              messages.append({"role": "user",
                               "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_fewshot}"}}]})
              messages.append({"role": "user", "content": self.user_prompt[i][1]})
             
            messages.append({"role": "user",
                             "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                                         {"type": "text", "text": text_input}]})
              
          else:
              messages=[
               {"role": "system", "content": self.system_prompt},
               {"role": "user", "content": self.user_prompt},
               {"role": "user", "content": [{"type": "image_url",
                                             "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                             "type": "text", "text": text_input}]}
              ]
              
        response = self.client.chat.completions.create(
              model=self.model_type,
              messages=messages,
              temperature=0.0,
              max_tokens=self.max_tokens,
        )
                
        return response.choices[0].message.content

    @lpbench
    def translate_scenarios(self, qs) -> list:
        response = self.client.beta.chat.completions.parse(
          model=self.model_type,
          messages=[
           {"role": "system", "content": self.system_prompt},
           {"role": "user", "content": f"{self.user_prompt}: {qs}"}
          ],
          response_format=ScenarioList
        )
                
        return response.choices[0].message.parsed

    @lpbench
    def translate_deepcontexts(self, q) -> list:
        response = self.client.chat.completions.create(
          model=self.model_type,
          messages=[
           {"role": "system", "content": self.system_prompt},
           {"role": "user", "content": f"{self.user_prompt}: {q}"}
          ],
          temperature=0.0,
        )
                
        return response.choices[0].message.content