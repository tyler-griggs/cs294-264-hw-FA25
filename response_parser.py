class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
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

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        if not isinstance(text, str):
            raise ValueError("Expected `text` to be a string")

        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            raise ValueError("Missing END_CALL marker")

        begin_idx = text.rfind(self.BEGIN_CALL, 0, end_idx)
        if begin_idx == -1:
            raise ValueError("Missing BEGIN_CALL marker before END_CALL")

        thought = text[:begin_idx].rstrip()
        block = text[begin_idx + len(self.BEGIN_CALL):end_idx].strip()
        if not block:
            raise ValueError("Empty function call block")

        lines = [ln.rstrip("\n") for ln in block.splitlines()]
        if not lines or not lines[0].strip():
            raise ValueError("Missing function name")
        func_name = lines[0].strip()

        args = {}
        i = 1
        while i < len(lines):
            if lines[i].strip() == self.ARG_SEP:
                i += 1
                if i >= len(lines):
                    raise ValueError("ARG_SEP without argument name")
                arg_name = lines[i].strip()
                if not arg_name:
                    raise ValueError("Empty argument name")
                i += 1
                # Collect value lines until next ARG_SEP or end-of-block
                val_start = i
                while i < len(lines) and lines[i].strip() != self.ARG_SEP:
                    i += 1
                arg_val = "\n".join(lines[val_start:i]).rstrip()
                args[arg_name] = arg_val
            else:
                # Ignore any stray lines outside ARG_SEP blocks
                i += 1

        return {"thought": thought, "name": func_name, "arguments": args}
