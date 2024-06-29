import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    # def generate_prompt(
    #     self,
    #     instruction: str,
    #     input: Union[None, str] = None,
    #     options: Union[None, str] = None,
    #     label: Union[None, str] = None,
    # ) -> str:
    #     # returns the full prompt from instruction and optional input
    #     # if a label (=response, =output) is provided, it's also appended.
    #     if options:
    #         res = self.template["prompt_input"].format(
    #             instruction=instruction, input=input, options=options
    #         )
    #     else:
    #         res = self.template["prompt_no_options"].format(
    #             instruction=instruction, input=input
    #         )
    #     if label:
    #         res = f"{res}{label}"
    #     if self._verbose:
    #         print(res)
    #     return res

    # def generate_prompt(self, input_field: str, output_field: Union[None, str] = None) -> str:
    #     # Create a new list from the template to avoid modifying the original
    #     prompt_list = []
    #     for item in self.template["prompt_input"]:
    #         # Assume the item content is a string that may require formatting
    #         new_item = {
    #             "role": item["role"],
    #             "content": item["content"].format(input_field=input_field)
    #         }
    #         prompt_list.append(new_item)

    #     # Append output_field if provided
    #     if output_field:
    #         output_prompt = {
    #             "role": "assistant",
    #             "content": output_field
    #         }
    #         prompt_list.append(output_prompt)

    #     # Convert each dictionary in the list to string and join them with commas
    #     result_prompt = ','.join([str(item) for item in prompt_list])

    #     # If verbose, print the prompt
    #     if self._verbose:
    #         print(result_prompt)

    #     return result_prompt

    # def generate_prompt(self, input_field: str, output_field: Union[None, str] = None) -> str:
    #     res = [
    #         {"role": "system", "content": self.template["system_prompt"]},
    #         {"role": "user", "content": input_field},
    #     ]

    #     if output_field:
    #         res.append({"role": "assistant", "content": output_field})
    #     if self._verbose:
    #         print(res)
    #     return res
    def generate_prompt(
        self,
        instruction: str,
        options: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:

        ############ 生成prompt_chat版
        res = [
            {"role": "system", "content": self.template["system_prompt"]},
            {"role": "user", "content": instruction},
        ]

        if label:
            res.append({"role": "assistant", "content": label})
        if self._verbose:
            print(res)
        return res


    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
