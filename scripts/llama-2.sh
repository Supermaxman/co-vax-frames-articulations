#!/bin/bash
python code/articulate.py --api deepinfra --api_key $DEEPINFRA_TOKEN --method few --model llama-2
python code/relations.py --api deepinfra --api_key $DEEPINFRA_TOKEN --method few --model llama-2
python code/articulate.py --api deepinfra --api_key $DEEPINFRA_TOKEN --method iccl --model llama-2
python code/relations.py --api deepinfra --api_key $DEEPINFRA_TOKEN --method iccl --model llama-2
