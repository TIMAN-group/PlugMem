# Use VLLM to host the model locally, modify parameters as needed
# -dp: number of devices to use
# -tp: number of threads to use
# > vllm.log: redirect the output to a file
# 2>&1: redirect the error to the same file
# &: run the command in the background

# Example:
# nohup vllm serve --model CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic -dp 4 -tp 1 > vllm.log 2>&1 &

nohup vllm serve --model CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic -dp 4 -tp 1 > vllm.log 2>&1 &