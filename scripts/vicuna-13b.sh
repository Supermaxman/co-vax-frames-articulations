#!/bin/bash
python code/articulate.py --api replicate --api_key $REPLICATE_API_TOKEN --method few --model vicuna-13b
python code/relations.py --api replicate --api_key $REPLICATE_API_TOKEN --method few --model vicuna-13b
python code/articulate.py --api replicate --api_key $REPLICATE_API_TOKEN --method iccl --model vicuna-13b
python code/relations.py --api replicate --api_key $REPLICATE_API_TOKEN --method iccl --model vicuna-13b
