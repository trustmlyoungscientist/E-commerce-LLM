import os
import random
from typing import Any, Dict, List

import vllm

from .base_model import ShopBenchBaseModel

#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))




class EclmPretrained(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = "deanpp/E-Commerce_LLM"

        # initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.4, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. "


    def get_batch_size(self) -> int:

        self.batch_size = 8
        return self.batch_size


    def format_prommpts(self, prompts):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        formatted_prompts = []
        for prompt in prompts:
            prompts_meg = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}]
            single_input = self.tokenizer.apply_chat_template(
                prompts_meg,
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted_prompts.append(single_input)

        return formatted_prompts


    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.
    
            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        prompts = batch["prompt"]
        formatted_prompts = self.format_prommpts(prompts)
        max_new_tokens = 100 
        
        if is_multiple_choice:
            max_new_tokens = 3 # For MCQ tasks, we only need to generate 1 token
        
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            sampling_params = vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm = False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        
        print(batch_response,"******************")

        return batch_response
