#!/bin/bash
# python code/articulate.py --api openai --api_key $OPENAI_KEY --method few --model gpt-3.5-turbo
python code/relations.py --api openai --api_key $OPENAI_KEY --method few --model gpt-3.5-turbo
python code/articulate.py --api openai --api_key $OPENAI_KEY --method iccl --model gpt-3.5-turbo
python code/relations.py --api openai --api_key $OPENAI_KEY --method iccl --model gpt-3.5-turbo
