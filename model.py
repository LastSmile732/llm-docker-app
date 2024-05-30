# model.py
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Logger configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

#model_path = "/opt/Llama-2-13B-chat-GPTQ"

class Model:
    def __init__(self, model_path):
        self.model_name = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self, precision='fp16'):
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise EnvironmentError("CUDA not available.")
            # Set precision settings
            if precision == 'fp16':
                torch_dtype = torch.float16
            elif precision == 'qint8':
                torch_dtype = torch.qint8
            else:
                torch_dtype = torch.float32
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set up model configuration
            config = AutoConfig.from_pretrained(self.model_name)

            #config.quantization_config["disable_exllama"] = False
            #config.quantization_config["use_exllama"] = True
            #config.quantization_config["exllama_config"] = {"version": 2}
            
            # Load model with configuration and precision
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                config=config, 
                device_map="cuda:0",  # Set to GPU 0
                torch_dtype=torch_dtype
            )

            self.loaded = True
            logger.info(f"Model loaded successfully on GPU with {precision} precision.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")   

    def predict(self, input_text, temperature=0.1, max_length=6000):
        if not self.loaded:
            logger.error("Model not loaded. Please load the model before prediction.")
            return None

        logger.info("========== Start Prediction ==========")
        try:
            # Ensure the input_text is a string
            if not isinstance(input_text, str):
                raise ValueError("Input text must be a string.")

            # Encoding the input text
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

            # Move input to the same device as model
            input_ids = input_ids.to(next(self.model.parameters()).device)

            # Generating output using the model
            outputs = self.model.generate(input_ids, max_length=max_length, repetition_penalty=1.2, num_return_sequences=1)
            
            # Decoding and returning the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Response: {}".format(response))
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            response = None

        logger.info("========== End Prediction ==========")
        return response