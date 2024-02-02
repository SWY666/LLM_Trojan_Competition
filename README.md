## Instructions

Run main.py directly and you can get an on-going result of the backdoor suffix search in the backdoored llama 1.

## Notes
1. We cannot use additional dataset.
2. Their backdoor effect is ``weak''.

## Current Reverse Engineering Strategy
1. First, we sift out those prompts with low reward values (from the provided dataset)