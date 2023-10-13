from abc import ABC, abstractmethod
import os
import time

import openai
import json


class ChatAPI(ABC):
    def __init__(self, api_key: str = None, cache_path: str = None):
        self.api_key = api_key
        self.cache_path = cache_path

    @abstractmethod
    def send(self, messages):
        pass

    @abstractmethod
    def build_message(self, text: str):
        pass


class OpenAIAPI(ChatAPI):
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: int = 512,
        delay_seconds: int = 6,
        api_key: str = None,
        cache_path: str = None,
    ):
        super().__init__(api_key, cache_path)
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_seconds = delay_seconds

    def send(self, messages):
        # check to see if we have a cached api response
        hash_key = str(hash(json.dumps(messages, sort_keys=True)))
        if self.cache_path is not None:
            cache_file = os.path.join(self.cache_path, f"{hash_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    api_response = json.load(f)
                response = self.process_response(api_response)
                return response
        while True:
            try:
                api_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                # rate limit requests
                time.sleep(self.delay_seconds)
                break
            except Exception as e:
                print(e)
                # rate limit errors more aggressively
                time.sleep(max(self.delay_seconds * 5, 30))
        if self.cache_path is not None:
            cache_file = os.path.join(self.cache_path, f"{hash_key}.json")
            with open(cache_file, "w") as f:
                json.dump(api_response, f)
        response = self.process_response(api_response)
        return response

    def build_message(self, text: str):
        return {"role": "user", "content": text}

    def process_response(self, response):
        message = {
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"],
        }
        return message
