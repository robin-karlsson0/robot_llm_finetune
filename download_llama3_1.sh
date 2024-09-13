#!/bin/bash

tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir ./models/llama_3_1_8B_instruct \
    --ignore-patterns "original/consolidated.*" \
    --hf-token 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