from eval_and_test import EVALUATEandTEST
from run import lora_model

instruction = ""
input = ""

print(EVALUATEandTEST.test(lora_model, instruction = instruction, input = input))
