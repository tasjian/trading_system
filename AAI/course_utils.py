from time import sleep

def not_implemented_gen(state):
    for letter in "Chain Not Implemented. Enter with no inputs or interrupt execution to exit":
        yield letter
        sleep(0.005)

def print_history(state):
    if state.get("messages"):
        for role, msg in state.get("messages"):
            if role == "ai": print("[Agent]: ", end="")
            if role == "user": print("[User]: ", end="")
            for letter in msg: 
                print(letter, end="")
                sleep(0.005)
            print()

def chat_with_chain(state={}, chain=not_implemented_gen):
    assert isinstance(state, dict)
    state["messages"] = state.get("messages", [])
    print_history(state)
    while True:
        try:
            human_msg = input("\n[Human]:")
            if not human_msg.strip():
                break
            agent_msg = ""
            state["messages"] += [("user", human_msg)]
            print("\n[Agent]: ", end="")
            for token in getattr(chain, "stream", chain)(state):
                agent_msg += token
                print(str(getattr(token, "content", token)), end="", flush=True)
            state["messages"] += [("ai", agent_msg)]
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break

from langchain_core.runnables import RunnableLambda
from tqdm.auto import tqdm
import threading

def batch_process(fn, inputs, max_concurrency=20):
    lock = threading.Lock()
    pbar = tqdm(total=len(inputs))
    def process_doc(value):
        try:
            output = fn(value)
        except Exception as e: 
            print(f"Exception in thread: {e}")
        with lock:
            pbar.update(1)
        return output
    try:
        lc_runnable = fn if hasattr(fn, "batch") else RunnableLambda(process_doc)
        return lc_runnable.batch(inputs, config={"max_concurrency": max_concurrency})
    finally:
        pbar.close()


####################################################
## Section 2

from langchain_core.output_parsers import PydanticOutputParser

def get_schema_hint(schema_obj):
    return PydanticOutputParser(pydantic_object=schema_obj).get_format_instructions().replace("{", "{{").replace("}", "}}")

SCHEMA_HINT = (
    'The output should be formatted as a JSON instance that conforms to the JSON schema below.'
    '\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description":'
    ' "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}'
    '\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.'
    ' The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.'
    '\n\nHere is the output schema:\n'
).replace("{", "{{").replace("}", "}}") + '```\n{schema_hint}\n```'

####################################################
## Section 3

from langgraph.types import Command
from colorama import Fore, Style
from copy import deepcopy
import json
import requests

def serialize_for_js(data):
    """
    Converts a Python dictionary to a clean JSON string that JavaScript can safely parse.
    """
    def custom_serializer(obj):
        """ Handle non-serializable objects like sets, tuples, and custom classes. """
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        if isinstance(obj, tuple):
            return [custom_serializer(item) for item in obj]  # Convert tuples recursively
        if hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        return str(obj)  # Fallback: Convert unknown objects to string

    return json.dumps(data, default=custom_serializer, ensure_ascii=False, indent=2)

# Helper function to send update to the web service
def send_update(mode, data):
    try:
        requests.post("http://lg_viz:3002/api/update", json={"mode": mode, "data": serialize_for_js(data)})
    except Exception as e:
        print(f"Error sending update to web interface: {e}")

def send_clear():
    try:
        requests.post("http://lg_viz:3002/api/clear", json={})
    except Exception as e:
        print(f"Error sending clear to web interface: {e}")

def stream_from_app(app_stream, input_buffer=[{"messages": []}], verbose=False, debug=False):
    seen_metas = dict()
    input_buffer = deepcopy(input_buffer)
    send_clear()
    while input_buffer:
        for mode, chunk in app_stream(input_buffer.pop(), stream_mode = ["values", "messages", "updates", "debug"]): 
            # https://langchain-ai.github.io/langgraph/concepts/streaming/
            if mode == "messages":
                chunk, meta = chunk
                if meta.get("checkpoint_ns") not in seen_metas:
                    caller_node = meta.get('langgraph_node')
                    user_prompt = f"node:{caller_node} -> message" if verbose else caller_node.title()
                    send_update("messages", {"meta": meta})
                    if verbose: 
                        print(f"[node:{meta.get('langgraph_node')}:meta -> message] {meta}", flush=True)
                    yield f"[{user_prompt}]: "
                seen_metas[meta.get("checkpoint_ns")] = meta
                if chunk.content:
                    # print(chunk.content, end="", flush=True)
                    yield chunk.content
                elif chunk.response_metadata:
                    if verbose: 
                        print(f"\n\n[message] {chunk.response_metadata=}, {chunk.usage_metadata=}", flush=True)
            elif mode == "values":
                if verbose: 
                    print("[value]", chunk, flush=True)
                send_update("values", chunk)
            elif mode == "updates":
                if verbose:
                    print("[update]", chunk, flush=True)
                send_update("updates", chunk)
                ## Handle the interrupt. If an interrupt happens, then handle the interrupt and re-queue app stream
                if "__interrupt__" in chunk and chunk.get("__interrupt__")[0].resumable:
                    user_prompt = "\n[update -> interrupt] " * bool(verbose) + chunk.get("__interrupt__")[0].value
                    input_buffer += [
                        Command(resume=input(user_prompt))
                    ]
            elif mode == "debug":
                send_update("debug", chunk)
                if debug:
                    print(Fore.RED + f"[debug] {chunk}" + Style.RESET_ALL)

