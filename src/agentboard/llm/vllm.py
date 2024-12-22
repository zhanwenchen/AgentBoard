from vllm import LLM, SamplingParams
from agentboard.common.registry import registry
from agentboard.prompts.prompt_template import prompt_templates
from agentboard.utils.logging.agent_logger import AgentLogger


@registry.register_llm("vllm")
class VLLM:
    def __init__(self,
                 model='',
                 temperature=0,
                 max_tokens=100,
                 top_p=1.0,
                 context_length=4096,
                 stop='\n',
                 ngpu=4,
                 d_type='bfloat16'
                 ):
        self.engine = self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.context_length = context_length
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens
        )
        if self.context_length > 8192:
            llm = LLM(model=str(model), dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.9, max_num_batched_tokens=8192, max_model_len=8192, enable_lora=True)
        else:
            # breakpoint()
            llm = LLM(model=str(model), dtype=d_type, tensor_parallel_size=ngpu, gpu_memory_utilization=0.9, max_num_batched_tokens=8192, max_model_len=8192, enable_lora=True)
        self.tokenizer = llm.get_tokenizer()
        self.model_str = model_str = model.lower()
        self.is_vicuna = 'vicuna' in model_str
        self.llm_generate = llm.generate
        self.llm = llm
        logger = AgentLogger(__name__)
        self.logger_info = logger.info
        if "codellama-13b" in model_str:
            full_prompt = prompt_templates["codellama-13b"]
        elif "codellama-34b" in model_str:
            full_prompt = prompt_templates["codellama-34b"]
        elif "llama" in model_str:
            full_prompt = prompt_templates["llama"]
        elif 'lemur' in model_str:
            full_prompt = prompt_templates["lemur"]
        elif 'vicuna' in model_str:
            full_prompt = prompt_templates["vicuna"]
        elif 'deepseek' in model_str:
            full_prompt = prompt_templates["deepseek"]
        elif 'mistral' in model_str:
            full_prompt = prompt_templates["mistral"]
        else:
            raise NotImplementedError
        self.full_prompt_format = full_prompt.format

    def make_prompt(self, system_message: str, prompt: str) -> str:
        system_message += "Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. \n"

        return self.full_prompt_format(system_prompt=system_message, prompt=prompt).strip()

    def generate(self, system_message, prompt, use_tqdm=False):
        full_prompt = self.make_prompt(system_message, prompt)
        assert full_prompt is not None
        outputs = self.llm_generate([full_prompt], self.sampling_params, use_tqdm=use_tqdm)
        outputs = outputs[0].outputs[0].text

        if self.is_vicuna is True:
            # Note: vicuna tends to generate get\_search\_movie with Action Input: {"movie\_name": "Crouching Tiger, Hidden Dragon"} when using tools
            outputs = outputs.replace(r'\_', '_')
        self.logger_info(f"Model {self.model_str}.generate: full_prompt={full_prompt}\n\noutputs={outputs}")
        return True, outputs

    def num_tokens_from_messages(self, messages):
        prompt = messages[1]["content"]
        system_message = messages[0]["content"]
        full_prompt = self.make_prompt(system_message, prompt)
        tokens = self.tokenizer(full_prompt)
        num_tokens = len(tokens["input_ids"])
        #print(num_tokens)
        return num_tokens

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        context_length = config.get("context_length", 4096)
        ngpu = config.get("ngpu", 4)
        dtype = config.get("dtype", 'bfloat16')
        return cls(model=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   top_p=top_p,
                   context_length=context_length,
                   stop=stop,
                   ngpu=ngpu,
                   d_type=dtype)

    def __del__(self):
        del self.llm.llm_engine.model_executor
        del self.llm
        del self.model, self.tokenizer
        del self
