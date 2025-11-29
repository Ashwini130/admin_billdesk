import sys

from groq import Groq

from commons.llm_utils import LLMUtils
from commons.utils import Utils

## Run command : python bill_extractor_tesseract.py D:/pycharm/admin_billdesk/resources/commute D:\pycharm\admin_billdesk\src\prompt\system_prompt_cab.txt
## export api key via PS :$env:GROQ_API_KEY="API_KEY"
class Extractor:

    path = sys.argv[1] #"D:/pycharm/admin_billdesk/resources/commute"
    system_prompt_file_path = sys.argv[2]

    receipts = Utils.process_folder(path)
    print(receipts)

    client = Groq()

    system_prompt = Utils.load_text_file(system_prompt_file_path)
    user_prompt = f"""{receipts}"""
    model = "llama-3.1-8b-instant"

    output = LLMUtils.call_llm(client,model,system_prompt,user_prompt,0)
    #print(output)
    Utils.write_json_to_file(output, "rides.json")