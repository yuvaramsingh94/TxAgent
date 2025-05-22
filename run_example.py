import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"
sys.path.append(os.path.join("src", "txagent"))
from utils import gpu_checker
from txagent import TxAgent
import gc

print("Check GPU")
gpu_checker()
if __name__ == "__main__":

    os.environ["MKL_THREADING_LAYER"] = "GNU"

    model_name = "mims-harvard/TxAgent-T1-Llama-3.1-8B"  # "mradermacher/TxAgent-T1-Llama-3.1-8B-GGUF"
    rag_model_name = "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
    multiagent = False
    max_round = 20
    init_rag_num = 0
    step_rag_num = 10

    agent = TxAgent(model_name, rag_model_name, enable_summary=True)
    agent.init_model()
    question = "A 20 year old male is suffering from fever. what treatment can be provided to him given he is afride of oral tablets. We found Ibuprofen is not workign with the patient. Help or he might die"
    # question = "A 30-year-old patient is taking Prozac to treat their depression. They were recently diagnosed with WHIM syndrome and require a treatment for that condition as well. Is Xolremdi suitable for this patient, considering contraindications?"
    # question = "Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of moderate hepatic impairment?"
    print("++++++++++++++++++++++++++")
    print("The Question:", question)

    response = agent.run_multistep_agent(
        question,
        temperature=0.3,
        max_new_tokens=1024,
        max_token=90240,
        call_agent=multiagent,
        max_round=max_round,
    )

    print(f"\033[94m{response}\033[0m")
    print("++++++++++++++++++++++++++")
    gc.collect()
