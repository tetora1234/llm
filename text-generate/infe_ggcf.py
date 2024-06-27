from llama_cpp import Llama
import time

CMAKE_ARGS="-DLLAMA_CUDA=on"
FORCE_CMAKE=1

llm = Llama(
    model_path="./text-generate/Ninja-v1-NSFW-128k-Q4_K_S.gguf",
    n_ctx=32768,
    use_gpu=True,
    n_gpu_layers=32
)

initial_prompt = """男性向けアダルトASMRの台本を作成してください。以下の指示に厳密に従ってください

タイトル：ボクっ娘幼馴染がアニコス姿でパイズリ、正常位中出しセックス

台詞の例：
「ねぇ、ボクのコスプレ、どう？ エッチだと思う？」
「ボクの胸で気持ちよくなってね...」
「あぁん...奥まで届いてる...」

それでは、台本を始めてください：
"""

def generate_text(prompt, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            generated_text = ""
            for output in llm.create_completion(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.8,
                top_p=0.95,
                top_k=100,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                repeat_penalty=1.1,
                stream=True
            ):
                chunk = output['choices'][0]['text']
                generated_text += chunk
                print(chunk, end='', flush=True)
            
            print("\n")
            return generated_text
        except RuntimeError as e:
            if "prefix-match hit" in str(e):
                print(f"Prefix-match hit occurred. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
    
    print("Max retries reached. Unable to generate text.")
    return ""

while True:
    generated_text = generate_text(initial_prompt)
    if generated_text:
        initial_prompt += generated_text

    # コンテキストの長さを管理
    max_context_length = 10000
    if len(initial_prompt) > max_context_length:
        initial_prompt = initial_prompt[-max_context_length:]

    # ユーザーに継続するか尋ねる
    user_input = input("Continue generating? (y/n): ")
    if user_input.lower() != 'y':
        break