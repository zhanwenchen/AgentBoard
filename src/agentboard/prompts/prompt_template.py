prompt_templates = {
    "deepseek":
            """
                {system_prompt}

                USER: {prompt}<｜end▁of▁sentence｜>

                ASSISTANT:
            """,
    "codellama-13b":
            """
                <s>
                <<SYS>>
                {system_prompt}
                <</SYS>>
                [INST]{prompt}[/INST]
            """,
    "codellama-34b":
            """
                <s>[INST]{system_prompt}{prompt}[/INST]
            """,
    "llama":
            """
                <<SYS>>
                {system_prompt}
                <</SYS>>
                [INST]{prompt}[/INST]
            """,
    "llama3":
            """
                <|start_header_id|>system
                {system_prompt}
                <|eot_id|>
                <|start_header_id|>user
                {prompt}
                <|eot_id|>
                <|start_header_id|>assistant
            """,
    "lemur":
            """
                <|im_start|>system
                {system_prompt}
                <|im_end|>
                <|im_start|>user
                {prompt}<|im_end|>
                <|im_start|>assistant\n
            """,
    "vicuna":
            """
            {system_prompt}

            USER: {prompt}</s>
            ASSISTANT:
            """,
    "mistral":
            """
            <s>
            {system_prompt}
            </s>
            [INST]{prompt}[/INST]
            """,
}

# _register_template(
#     name="llama2",
#     format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
#     format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
# )

# _register_template(
#     name="llama3",
#     format_user=StringFormatter(
#         slots=[
#             (
#                 "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
#                 "<|start_header_id|>assistant<|end_header_id|>\n\n"
#             )
#         ]
#     ),
#     format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
#     format_observation=StringFormatter(
#         slots=[
#             (
#                 "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
#                 "<|start_header_id|>assistant<|end_header_id|>\n\n"
#             )
#         ]
#     ),
#     format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
#     stop_words=["<|eot_id|>"],
#     replace_eos=True,
#     replace_jinja_template=False,
# )
