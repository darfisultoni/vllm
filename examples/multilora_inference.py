"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""
from time import time

from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest


def create_test_prompts(lora_path: str, lora_id: int) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters.
    
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
            ("explain bernoulli law",
         SamplingParams(
                        ignore_eos=True,
                        max_tokens=100,
                        ), 
         #None ),
         LoRARequest("qna-lora", 5*lora_id+i, lora_path)) for i in range(1,6)
    ] * 10


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    the_start = time()
    durations = {}
    real_durations = []

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            start_time = time()
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            durations[str(request_id)] = start_time
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                end_time = time()
                duration = end_time - durations[request_output.request_id]
                real_durations.append(duration)
                print(f"Request {request_output.request_id} completed in {duration:.4f} seconds:")
                print(len(request_output.outputs[0].token_ids))
    print(f"avg {sum(real_durations)/len(real_durations):.4f} seconds")
    print(f"total reqs {len(real_durations):.0f} reqs")
    print(f"total time {time()-the_start:.4f} seconds")


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="huggyllama/llama-7b",
                             enable_lora=True,
                             max_loras=10,
                             max_lora_rank=64,
                             max_cpu_loras=100,
                             max_num_seqs=48)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = snapshot_download(repo_id="MBZUAI/bactrian-x-llama-7b-lora")
    lora_path2 = snapshot_download(repo_id="tloen/alpaca-lora-7b")
    test_prompts = create_test_prompts(lora_path, 1)
    test_prompts2 = create_test_prompts(lora_path2, 2)
    #process_requests(engine, test_prompts)
    process_requests(engine, test_prompts + test_prompts2)


if __name__ == '__main__':
    main()
