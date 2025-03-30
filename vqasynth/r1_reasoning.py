import time
import base64
import io
import random
from PIL import Image
from openai import OpenAI

class R1Reasoner:
    def __init__(
        self,
        api_key: str,
        model: str,
        image_column: str,
        text_column: str,
        delay: int = 1
    ):
        """
        Args:
            api_key: API key for OpenAI (used by your custom OpenAI client or official library).
            model: Model name (e.g., 'gpt-4').
            image_column: The name of the column where images are stored.
            text_column: The name of the column containing conversation data (list of dicts).
            delay: Minimum time (seconds) to wait between calls.
        """
        self.model = model
        self.image_column = image_column
        self.text_column = text_column
        self.delay = delay

        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image):
        """
        Encodes an image as base64. Handles either a file path (str) or a PIL.Image.
        Returns a base64-encoded string.
        """
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return encoded
        return ""

    def _extract_qa_pairs(self, conversation):
        """
        Extract all user->assistant Q/A pairs from the conversation.
        Returns a list of (question_str, answer_str) tuples.
        """
        pairs = []
        for i in range(len(conversation) - 1):
            current_msg = conversation[i]
            next_msg = conversation[i + 1]
            # Check if current is user, next is assistant
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                question_texts = [
                    c.get("text", "") for c in current_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                answer_texts = [
                    c.get("text", "") for c in next_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                if question_texts and answer_texts:
                    # Take the first text if multiple appear
                    pairs.append((question_texts[0], answer_texts[0]))
        return pairs

    def _find_random_qa_pair(self, conversation):
        """
        Finds a random user->assistant Q/A pair from the conversation.
        Returns (question_str, answer_str) or (None, None) if none found.
        """
        pairs = self._extract_qa_pairs(conversation)
        if not pairs:
            return None, None
        qa = random.choice(pairs)
        return qa

    def _format_input_string(self, question):
        """
        Format the question with special tokens into your desired string for 'input'.
        Example format:
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>  detailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>  {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        input_str = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>  detailed thinking on"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>  "
            f"{question}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return input_str

    def run(self, question: str, answer: str, image):
        """
        Build a prompt that includes the question/answer pair,
        call the model, and return the chain-of-thought or reasoning.

        Args:
            question: The question from user
            answer: The assistant's answer
            image: The image data (path or PIL image)
        Returns:
            reasoning (str)
        """
        # Encode image
        base64_image = self.encode_image(image)

        # Example system prompt
        system_prompt = (
            "Please see the quantitative distance question-answer pair given and use the remaining information "
            "to formulate a reasoning trace to support the answer. Some information may be partially incorrect, "
            "so you will need to reason about the scene and its scale to ensure high-quality, consistent, and robust responses. "
            "Make sure to generate a CoT trace in the first-person voice, as though revealing an internal monologue, "
            "inside of <think> tokens before providing the final answer inside <answer> tokens. Don't refer to the information "
            "as if it were stated, mentioned, previously referenced, or provided, but rather inferred or directly observed as if "
            "you were an embodied AI tasked with making these judgements after some thought about what you see in the scene. "
            "Feel free to simplify the way objects in the scene are referenced for a more colloquial description.\n"
        )

        qa_pair = f"Question: {question} Answer: {answer}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt + qa_pair},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ]
        )

        reasoning = response.choices[0].message.content
        return reasoning

    def apply_transform(self, examples, images=None, text=None):
        """
        Called by dataset.map(..., batched=True).
        """
        images = images or self.image_column
        text = text or self.text_column

        is_batched = isinstance(examples[images], list)

        if not is_batched:
            # Single example scenario
            conversation = examples[text]
            image_data = examples[images]
            if isinstance(image_data, list) and len(image_data) > 0:
                image_data = image_data[0]

            question, ans = self._find_random_qa_pair(conversation)

            if question is None or ans is None:
                examples["input"] = None
                examples["output"] = None
                examples["reasoning"] = None
            else:
                input_string = self._format_input_string(question)
                examples["input"] = input_string
                reasoning_str = self.run(question, ans, image_data)
                examples["output"] = reasoning_str
                examples["reasoning"] = "on"

            return examples

        else:
            conversation_batch = examples[text]
            image_batch = examples[images]

            inputs = []
            outputs = []
            reasonings = []

            for idx, (conv, img) in enumerate(zip(conversation_batch, image_batch)):
                if isinstance(img, list) and len(img) > 0:
                    img = img[0]

                question, ans = self._find_random_qa_pair(conv)

                if question is None or ans is None:
                    inputs.append(None)
                    outputs.append(None)
                    reasonings.append(None)
                    continue

                # Create the formatted input
                input_string = self._format_input_string(question)
                inputs.append(input_string)

                # Get reasoning from the model
                reasoning_str = self.run(question, ans, img)
                outputs.append(reasoning_str)
                reasonings.append("on")

            examples["input"] = inputs
            examples["output"] = outputs
            examples["reasoning"] = reasonings

            return examples
