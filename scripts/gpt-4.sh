#!/bin/bash
python code/articulate.py --api openai --api_key $OPENAI_KEY --method few --model gpt-4
python code/relations.py --api openai --api_key $OPENAI_KEY --method few --model gpt-4
