import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import asyncio
from semantic_kernel.planning.basic_planner import BasicPlanner
from fastapi import FastAPI
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.core_plugins import (
    FileIOPlugin,
    MathPlugin,
    TextPlugin,
    TimePlugin,
)
import logging


logging.basicConfig(filename="logger.log",level=logging.INFO)

app=FastAPI()

kernel = sk.Kernel()

api_key=""
kernel.add_chat_service("gpt-4",OpenAIChatCompletion(ai_model_id="gpt-3.5-turbo-1106", api_key=api_key))
    
plugins_directory = ""
receipe_gener = kernel.import_semantic_plugin_from_directory(plugins_directory, "cooking")
summarize_plugin=kernel.import_semantic_plugin_from_directory(plugins_directory, "WriterPlugin")
kernel.import_plugin(MathPlugin(), "math")

@app.get("/basic/{name}")
async def main(name):

    planner = BasicPlanner()
    ask = f"make a receipe on {name} and summarize the output "
    new_plan = await planner.create_plan(ask, kernel)
    print(new_plan.generated_plan)
    results = await planner.execute_plan(new_plan, kernel)
    return str(results).replace("\n", "")

@app.get("/sequential/{name}")
async def seqq(name):
    ask = f"make a receipe on {name} and summarize the output "
    planner = SequentialPlanner(kernel)
    sequential_plan = await planner.create_plan(goal=ask)

    for step in sequential_plan._steps:
        print(step.description, ":", step._state.__dict__)

    result = await sequential_plan.invoke()
    return str(result).replace("\n", "")

@app.get("/action/{name}")
async def action(name):
    ask = f"generate a java program on {name}? "
    planner = ActionPlanner(kernel)
    '''
    kernel.import_plugin(FileIOPlugin(), "fileIO")
    kernel.import_plugin(TimePlugin(), "time")
    kernel.import_plugin(TextPlugin(), "text")
    '''
    plan = await planner.create_plan(goal=ask)

    result = await plan.invoke()
    return str(result).replace("\n", "")



if __name__=="__main__":
    asyncio.run(main())
    asyncio.run(action())
    asyncio.run(seqq())

