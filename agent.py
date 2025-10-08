"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any
from datetime import datetime

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

SYSTEM_PROMPT = """You are a software engineering agent that solves coding tasks.

WORKFLOW:
1. Use run_bash_cmd to explore the codebase and understand the problem
2. Use run_bash_cmd to make file modifications (with sed, cat with heredoc, echo, etc.)
3. Use run_bash_cmd to test your changes
4. Once all changes are complete, call finish() with a brief summary

IMPORTANT: You must make actual file changes using bash commands. After you call finish(), 
the system will automatically run 'git add -A && git diff --cached' to capture your changes 
as a patch. Do NOT include the patch in your finish() call - just provide a brief summary 
of what you changed.


You must follow the following format for your responses:

your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
arg2_value (can be multiline)
...
{END_CALL}
"""

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Create root/system → user → instructor chain
        self.system_message_id = self.add_message("system", SYSTEM_PROMPT)
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message("instructor", "")

        # Register built-ins
        self.add_functions([self.finish, self.add_instructions_and_backtrack])

    # -------------------- MESSAGE TREE --------------------
    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        msg_id = len(self.id_to_message)
        parent_id = self.current_message_id if self.root_message_id != -1 else None

        msg = {
            "unique_id": msg_id,
            "role": role,
            "content": content,
            "timestamp": self._now(),
            "parent": parent_id,
            "children": [],
        }
        self.id_to_message.append(msg)

        if self.root_message_id == -1:
            self.root_message_id = msg_id
        else:
            # Link into parent's children
            if parent_id is not None:
                self.id_to_message[parent_id]["children"].append(msg_id)

        self.current_message_id = msg_id
        return msg_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        self.id_to_message[message_id]["content"] = content
        self.id_to_message[message_id]["timestamp"] = self._now()

    def _path_root_to(self, node_id: int) -> List[int]:
        path: List[int] = []
        cur = node_id
        while cur is not None and cur != -1:
            path.append(cur)
            cur = self.id_to_message[cur]["parent"]
        return list(reversed(path))

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        parts: List[str] = []
        for mid in self._path_root_to(self.current_message_id):
            parts.append(self.message_id_to_context(mid))
        return "".join(parts)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for fn in tools:
            if not callable(fn):
                continue
            self.function_map[fn.__name__] = fn  # minimal: last one wins on name collision

    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args:
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # Update the instruction node
        self.set_message_content(self.instructions_message_id, instructions)

        # Backtrack pointer (validate id range)
        if not (0 <= at_message_id < len(self.id_to_message)):
            raise ValueError(f"Invalid backtrack id: {at_message_id}")
        self.current_message_id = at_message_id
        return f"Backtracked to message {at_message_id} and updated instructions."

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps:
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        # Set/overwrite the user task
        self.set_message_content(self.user_message_id, task)

        last_tool_out = ""
        for step in range(max_steps):
            print(f"Step {step}")
            try:
                # 1) Reason: ask LLM with current context
                context = self.get_context()
                llm_text = self.llm.generate(context)

                # 2) Parse function call (thought + call)
                try:
                    parsed = self.parser.parse(llm_text)
                except ValueError as e:
                    # Parser failed - give feedback to LLM
                    error_msg = f"ERROR: Failed to parse your response. {str(e)}\nPlease follow the exact format specified."
                    self.add_message("assistant", llm_text)
                    self.add_message("tool", error_msg)
                    continue

                thought = parsed.get("thought", "")
                func_name = parsed.get("name", "")
                args = parsed.get("arguments", {}) or {}

                # Record the assistant's reasoning
                self.add_message("assistant", llm_text)

                # Coerce common integer ids if present (minimal convenience)
                if "at_message_id" in args:
                    try:
                        args["at_message_id"] = int(args["at_message_id"])
                    except Exception:
                        pass  # leave as-is if not convertible

                # 3) Act: run tool
                if func_name not in self.function_map:
                    error_msg = f"ERROR: Unknown tool '{func_name}'. Available tools: {', '.join(self.function_map.keys())}"
                    self.add_message("tool", error_msg)
                    continue

                tool_fn = self.function_map[func_name]
                
                try:
                    result = tool_fn(**args) if args else tool_fn()
                    last_tool_out = str(result)
                except TypeError as e:
                    # Wrong arguments passed to function
                    sig = inspect.signature(tool_fn)
                    error_msg = f"ERROR: Invalid arguments for '{func_name}'. {str(e)}\nExpected signature: {func_name}{sig}"
                    self.add_message("tool", error_msg)
                    continue
                except Exception as e:
                    # Tool execution failed
                    error_msg = f"ERROR: Tool '{func_name}' failed: {str(e)}"
                    self.add_message("tool", error_msg)
                    continue


                print(f"\n\nllm_text: {llm_text}")
                print(f"\n\nlast_tool_out: {last_tool_out}")

                # Append tool output
                self.add_message("tool", last_tool_out)

                # 4) Done?
                if func_name == "finish":
                    return last_tool_out

            except Exception as e:
                # Catch-all for unexpected errors
                error_msg = f"ERROR: Unexpected error at step {step}: {str(e)}"
                self.add_message("tool", error_msg)
                continue

        # If we get here, we did not finish in time - return last output gracefully
        # The generate_patch() function will still create a patch from git changes
        return last_tool_out if last_tool_out else ""

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"


def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)


if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()
