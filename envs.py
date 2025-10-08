from utils import get_sb_environment
import subprocess

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            result = self.env.execute(command)
            if result["returncode"] != 0:
                raise ValueError(result["output"])
            return result["output"]
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            patch_result = self.env.execute("git add -A && git diff --cached")
            if patch_result["returncode"] != 0:
                return f"{result}\n\nError running git commands: {patch_result['output']}"
            patch_output = patch_result["output"]
            print(f"\n\npatch_output: {patch_output}")
            stripped = patch_output.strip()
            print(f"\n\nstripped: {stripped} <END_STRIPPED>")
            if stripped:
                return stripped
            else:
                return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"
    
    def search_repo(self, pattern: str, path=".") -> str:
        """
        Search the repository for the given pattern using rg (ripgrep) if available, otherwise grep.
        Results are bounded to avoid overwhelming output.
        
        Args:
            pattern (str): the pattern to search for
            path (str): the path to search in (default: ".")
            
        Returns:
            The search results as a string
        """
        # Check if rg (ripgrep) is available
        check_rg = self.env.execute("command -v rg")
        
        if check_rg["returncode"] == 0:
            # Use ripgrep with bounded output
            # --max-count limits matches per file
            # --max-columns limits line length to avoid huge lines
            # -n shows line numbers
            # -H shows filenames
            cmd = f"rg --max-count 100 --max-columns 500 -n -H {self._quote_arg(pattern)} {self._quote_arg(path)}"
        else:
            # Fall back to grep with bounded output
            # -r for recursive
            # -n for line numbers
            # -H for filenames
            # -m limits matches per file
            # Using head to limit total output lines
            cmd = f"grep -r -n -H -m 100 {self._quote_arg(pattern)} {self._quote_arg(path)} 2>/dev/null | head -n 500"
        
        try:
            result = self.env.execute(cmd)
            # grep/rg return non-zero if no matches found, which is not an error
            if result["returncode"] == 0:
                return result["output"]
            elif result["returncode"] == 1:
                # No matches found
                return "No matches found."
            else:
                # Actual error
                return f"Search error: {result['output']}"
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _quote_arg(self, arg: str) -> str:
        """
        Safely quote an argument for shell execution.
        """
        # Escape single quotes and wrap in single quotes
        return f"'{arg.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
    
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace the content of the file from the given line to the given line with the given content
        """
        raise NotImplementedError("replace_in_file must be implemented by the student")
    
    def show_file(self, file_path: str) -> str:
        """
        [Optional]Show the content of the file
        """
        raise NotImplementedError("show_file must be implemented by the student")

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output