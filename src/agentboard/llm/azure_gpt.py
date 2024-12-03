from openai import AzureOpenAI
import tiktoken
import time
import timeout_decorator
from agentboard.common.registry import registry
from agentboard.utils.logging.agent_logger import AgentLogger


TIMEOUT = 10


@registry.register_llm("gpt_azure")
class OPENAI_GPT_AZURE:
    def __init__(self,
                 retry_delays, # in seconds
                 engine="gpt-35-turbo",
                 temperature=0,
                 max_tokens=100,
                 system_message="You are a helpful assistant.",
                 use_azure=True,
                 top_p=1,
                 stop='\n',
                 max_retry_iters=5,
                 context_length=4096,
        ):


        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.use_azure = use_azure
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        self.client_chat_completions_create = AzureOpenAI().chat.completions.create
        self.logger = AgentLogger(__name__)
        # self.init_api_key()


    # def init_api_key(self):
    #     if self.use_azure:
    #         # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.environ['OPENAI_API_BASE'])'
    #         # openai.api_base = os.environ['OPENAI_API_BASE']
    #     else:

    @timeout_decorator.timeout(TIMEOUT)
    def chat_inference(self, messages):
        assert self.engine in ["gpt-35-turbo", "gpt-4", "gpt-35-turbo-16k"], "engine not supported in ChatCompletion"

        response = self.client_chat_completions_create(model=self.engine, # engine = "deployment_name".
            messages=messages,
            stop = self.stop,
            temperature = self.temperature,
            max_tokens = self.max_tokens
        )
        content = response.choices[0].message.content
        self.logger.info(f'Input: messages={messages}.\nOutput: content={content}')
        return content



    @timeout_decorator.timeout(TIMEOUT)
    def completion_inference(self, prompt, num_return_sequences=1, use_beam_search=False, logprobs=5):
        assert self.engine in ["text-davinci-003"], "engine not supported in Completion"

        if use_beam_search:
            response = self.client_chat_completions_create(
                model=self.engine,
                prompt=prompt,
                logprobs=logprobs,
                n=num_return_sequences,
                temperature=self.temperature,
                stop=self.stop,
                max_tokens=self.max_tokens
            )

            text=[choice["text"] for choice in response.choices],
            log_prob=[choice["logprobs"] for choice in response.choices]

            return text, log_prob
        else:
            response = self.client_chat_completions_create(
                model=self.engine,
                prompt=prompt,
                stop=self.stop,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].text

    def generate(self, system_message, prompt):

        if 'gpt' in self.engine:

            prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            # output = self.chat_inference(prompt)
            # output = output.split("\n")[0]
            # return True, output # return success, completion
            for attempt in range(self.max_retry_iters):  
                try:
                    output = self.chat_inference(prompt)
                    output = output.split("\n")[0]
                    return True, output # return success, completion
                except Exception as e:
                    self.logger.error(f"Error on attempt {attempt + 1}. e={e}", exc_info=e)
                    if attempt < self.max_retry_iters - 1:  # If not the last attempt
                        time.sleep(self.retry_delays)  # Wait before retrying

                    else:
                        self.logger.error(f"Failed to get completion after {attempt} attempts. e={e}", exc_info=e)
                        # raise e

            return False, None

        elif 'text' in self.engine:

            prompt = f'system_message\n{prompt}'

            for attempt in range(self.max_retry_iters):  
                try:
                    output = self.completion_inference(prompt)
                    output = output.split("\n")[0]
                    return True, output # return success, completion
                except Exception as e:
                    self.logger.error(f"Error on attempt {attempt + 1}. e={e}")
                    if attempt < self.max_retry_iters - 1:  # If not the last attempt
                        time.sleep(self.retry_delays)  # Wait before retrying

                    else:
                        self.logger.error(f"Failed to get completion after {attempt} attempts. e={e}", exc_info=e)
                        raise e

            return False, None


    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        model = self.engine
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 0
        tokens_per_name = 0
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4",
            "gpt-35-turbo",
            }:
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @classmethod
    def from_config(cls, config):
        retry_delays = config["retry_delays"]
        engine = config["engine"] #, "gpt-35-turbo")
        temperature = config["temperature"] #, 0)
        max_tokens = config["max_tokens"] #, 100)
        system_message = config["system_message"] #, "You are a helpful assistant.")
        use_azure = config["use_azure"] #, True)
        top_p = config["top_p"] #, 1)
        stop = config["stop"] #, ["\n"])
        context_length = config["context_length"] #, 4096)
        return cls(retry_delays,
                   engine=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   system_message=system_message,
                   use_azure=use_azure,
                   context_length=context_length,
                   top_p=top_p,
                   stop=stop)
