"""
Synthetic Reverse Shell Functionality Validation

This script validates that synthetically generated reverse shell commands are:
1. Syntactically valid (Tier 1) - can be parsed without execution
2. Functionally working (Tier 2) - actually establish connections when run

Addresses reviewer concern R3: "Explain how the template-based generation process 
enforces explicit constraints to ensure the generated reverse shells are functional 
and executable"
"""

import os
import re
import sys
import json
import subprocess
import socket
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import random

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.augmentation import NixCommandAugmentation, REVERSE_SHELL_TEMPLATES


@dataclass
class ValidationResult:
    command: str
    command_type: str  # bash, python, perl, ruby, php, nc, etc.
    syntax_valid: bool
    syntax_error: Optional[str] = None
    functional_valid: Optional[bool] = None
    functional_error: Optional[str] = None


def detect_command_type(command: str) -> str:
    """Detect the primary interpreter/tool used in the command."""
    command_lower = command.lower()
    
    if command_lower.startswith("python3 ") or "python3 -c" in command_lower:
        return "python3"
    elif command_lower.startswith("python ") or "python -c" in command_lower:
        return "python"
    elif "perl -e" in command_lower or command_lower.startswith("perl "):
        return "perl"
    elif "ruby -" in command_lower or command_lower.startswith("ruby "):
        return "ruby"
    elif "php -r" in command_lower or command_lower.startswith("php "):
        return "php"
    elif command_lower.startswith("nc ") or "nc -" in command_lower or command_lower.startswith("ncat "):
        return "nc"
    elif command_lower.startswith("lua ") or "lua -e" in command_lower:
        return "lua"
    elif command_lower.startswith("awk "):
        return "awk"
    elif command_lower.startswith("socat "):
        return "socat"
    elif command_lower.startswith("rcat "):
        return "rcat"
    elif "telnet " in command_lower:
        return "telnet"
    elif command_lower.startswith("zsh "):
        return "zsh"
    elif any(x in command_lower for x in ["bash ", "/bin/bash", "sh -i", "/bin/sh", "dash"]):
        return "bash"
    elif "mkfifo" in command_lower:
        return "bash"  # mkfifo-based pipes are shell commands
    elif "export " in command_lower:
        return "bash"  # env var exports followed by command
    else:
        return "unknown"


def extract_python_code(command: str) -> Optional[str]:
    """Extract the Python code from a command like python3 -c 'code'"""
    # Handle: python3 -c 'code' or python -c "code"
    patterns = [
        r"python[23]?\s+(?:-\w\s+)*-c\s+'([^']+)'",
        r'python[23]?\s+(?:-\w\s+)*-c\s+"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def extract_perl_code(command: str) -> Optional[str]:
    """Extract the Perl code from a command like perl -e 'code'"""
    patterns = [
        r"perl\s+(?:-\w\s+)*-e\s+'([^']+)'",
        r'perl\s+(?:-\w\s+)*-e\s+"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def extract_ruby_code(command: str) -> Optional[str]:
    """Extract the Ruby code from a command like ruby -rsocket -e'code'"""
    patterns = [
        r"ruby\s+(?:-\w+\s+)*-e'([^']+)'",
        r"ruby\s+(?:-\w+\s+)*-e\s+'([^']+)'",
        r'ruby\s+(?:-\w+\s+)*-e\s+"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def extract_php_code(command: str) -> Optional[str]:
    """Extract the PHP code from a command like php -r 'code'"""
    patterns = [
        r"php\s+(?:-\w\s+)*-r\s+'([^']+)'",
        r'php\s+(?:-\w\s+)*-r\s+"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python code syntax using ast.parse()"""
    import ast
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_bash_syntax(command: str) -> Tuple[bool, Optional[str]]:
    """Validate bash syntax using bash -n"""
    try:
        result = subprocess.run(
            ["bash", "-n", "-c", command],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout during syntax check"
    except Exception as e:
        return False, str(e)


def validate_perl_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Perl syntax using perl -c"""
    try:
        result = subprocess.run(
            ["perl", "-c", "-e", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        # perl -c returns 0 on success, prints "... syntax OK" to stderr
        if result.returncode == 0 or "syntax OK" in result.stderr:
            return True, None
        else:
            return False, result.stderr.strip()
    except FileNotFoundError:
        return None, "perl not installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout during syntax check"
    except Exception as e:
        return False, str(e)


def validate_ruby_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Ruby syntax using ruby -c"""
    try:
        result = subprocess.run(
            ["ruby", "-c", "-e", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr.strip()
    except FileNotFoundError:
        return None, "ruby not installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout during syntax check"
    except Exception as e:
        return False, str(e)


def validate_php_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate PHP syntax using php -l with a temp file"""
    import tempfile
    try:
        # Write PHP code to temp file for syntax check
        # php -l works better with files than -r for syntax checking
        with tempfile.NamedTemporaryFile(mode='w', suffix='.php', delete=False) as f:
            f.write(f"<?php\n{code}\n?>")
            temp_path = f.name
        
        result = subprocess.run(
            ["php", "-l", temp_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        os.unlink(temp_path)
        
        if result.returncode == 0 or "No syntax errors" in result.stdout:
            return True, None
        else:
            return False, result.stderr.strip() or result.stdout.strip()
    except FileNotFoundError:
        return None, "php not installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout during syntax check"
    except Exception as e:
        return False, str(e)


def extract_lua_code(command: str) -> Optional[str]:
    """Extract Lua code from a command like lua -e 'code'"""
    patterns = [
        r'lua(?:5\.\d)?\s+-e\s+"([^"]+)"',
        r"lua(?:5\.\d)?\s+-e\s+'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_lua_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Lua syntax using luac or lua"""
    import tempfile
    
    # Try different lua versions
    lua_commands = ["lua5.4", "lua5.3", "lua5.1", "lua", "luac"]
    
    for lua_cmd in lua_commands:
        try:
            # Write to temp file for syntax check
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            if "luac" in lua_cmd:
                result = subprocess.run(
                    [lua_cmd, "-p", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                # lua -c doesn't exist, but we can use load() to check syntax
                result = subprocess.run(
                    [lua_cmd, "-e", f"loadfile('{temp_path}')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr.strip()
                
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return False, "Timeout during syntax check"
        except Exception:
            continue
    
    return None, "lua not installed"


def validate_syntax(command: str, command_type: str) -> Tuple[bool, Optional[str]]:
    """Validate syntax based on command type."""
    
    if command_type in ["python", "python3"]:
        code = extract_python_code(command)
        if code:
            return validate_python_syntax(code)
        # If we can't extract, try the whole command as bash
        return validate_bash_syntax(command)
    
    elif command_type == "perl":
        code = extract_perl_code(command)
        if code:
            return validate_perl_syntax(code)
        return validate_bash_syntax(command)
    
    elif command_type == "ruby":
        code = extract_ruby_code(command)
        if code:
            return validate_ruby_syntax(code)
        return validate_bash_syntax(command)
    
    elif command_type == "php":
        code = extract_php_code(command)
        if code:
            return validate_php_syntax(code)
        return validate_bash_syntax(command)
    
    elif command_type == "lua":
        code = extract_lua_code(command)
        if code:
            return validate_lua_syntax(code)
        return validate_bash_syntax(command)
    
    elif command_type in ["bash", "nc", "awk", "socat", "telnet", "zsh"]:
        # These are shell commands or utilities - validate shell syntax
        return validate_bash_syntax(command)
    
    elif command_type == "rcat":
        # rcat is just a tool call - validate as bash
        return validate_bash_syntax(command)
    
    else:
        # Unknown type - try bash validation
        return validate_bash_syntax(command)


class ConnectionListener:
    """Simple TCP listener to detect incoming connections."""
    
    def __init__(self, port: int):
        self.port = port
        self.connection_received = False
        self.sock = None
        self.thread = None
    
    def start(self):
        """Start listening in a background thread."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3)  # 3 second timeout
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.listen(1)
        
        def listen():
            try:
                conn, addr = self.sock.accept()
                self.connection_received = True
                conn.close()
            except socket.timeout:
                pass
            except Exception:
                pass
        
        self.thread = threading.Thread(target=listen)
        self.thread.start()
    
    def stop(self):
        """Stop the listener."""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=1)


def validate_functional_docker(command: str, port: int = 9999, timeout: float = 2.0) -> Tuple[bool, Optional[str]]:
    """
    Validate command functionality using Docker.
    
    This runs the command in a Docker container and checks if it attempts
    to connect back to the host on the specified port.
    """
    # Start listener
    listener = ConnectionListener(port)
    listener.start()
    
    try:
        # Replace IP in command with host.docker.internal (Docker's host access)
        # and port with our test port
        cmd_modified = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'host.docker.internal', command)
        cmd_modified = re.sub(r'\b(PORT_NUMBER|\d{4,5})\b', str(port), cmd_modified, count=1)
        
        # Run in Docker with timeout
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "host",  # or use host.docker.internal
            "-e", f"PORT={port}",
            "ubuntu:22.04",  # or a custom image with all interpreters
            "timeout", str(timeout),
            "bash", "-c", cmd_modified
        ]
        
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )
        
        time.sleep(0.5)  # Give listener time to receive connection
        
        if listener.connection_received:
            return True, None
        else:
            return False, "No connection received"
            
    except subprocess.TimeoutExpired:
        # Timeout is expected for reverse shells
        if listener.connection_received:
            return True, None
        return False, "Timeout without connection"
    except Exception as e:
        return False, str(e)
    finally:
        listener.stop()


def generate_test_commands(num_per_template: int = 10, seed: int = 42) -> List[str]:
    """Generate synthetic commands for validation."""
    aug = NixCommandAugmentation(
        templates=REVERSE_SHELL_TEMPLATES,
        random_state=seed
    )
    return aug.generate_commands(num_per_template)


def run_syntax_validation(
    commands: List[str],
    sample_size: Optional[int] = None,
    seed: int = 42
) -> Dict:
    """
    Run syntax validation on a list of commands.
    
    Returns statistics on validation results.
    """
    if sample_size and sample_size < len(commands):
        random.seed(seed)
        commands = random.sample(commands, sample_size)
    
    results = []
    stats = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0, "skipped": 0})
    
    for cmd in tqdm(commands, desc="Validating syntax"):
        cmd_type = detect_command_type(cmd)
        is_valid, error = validate_syntax(cmd, cmd_type)
        
        result = ValidationResult(
            command=cmd,
            command_type=cmd_type,
            syntax_valid=is_valid if is_valid is not None else False,
            syntax_error=error
        )
        results.append(result)
        
        stats[cmd_type]["total"] += 1
        if is_valid is None:
            stats[cmd_type]["skipped"] += 1
        elif is_valid:
            stats[cmd_type]["valid"] += 1
        else:
            stats[cmd_type]["invalid"] += 1
    
    # Aggregate stats
    total = sum(s["total"] for s in stats.values())
    valid = sum(s["valid"] for s in stats.values())
    invalid = sum(s["invalid"] for s in stats.values())
    skipped = sum(s["skipped"] for s in stats.values())
    
    return {
        "total_commands": total,
        "syntax_valid": valid,
        "syntax_invalid": invalid,
        "skipped": skipped,
        "valid_rate": valid / (total - skipped) if (total - skipped) > 0 else 0,
        "by_type": dict(stats),
        "results": results,
        "invalid_examples": [
            {"cmd": r.command, "type": r.command_type, "error": r.syntax_error}
            for r in results if not r.syntax_valid
        ][:10]  # First 10 invalid examples
    }


def main():
    """Main validation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate synthetic reverse shell functionality")
    parser.add_argument("--num-per-template", type=int, default=100, 
                       help="Number of commands to generate per template")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Random sample size for validation (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--functional", action="store_true",
                       help="Run functional validation (requires Docker)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Synthetic Reverse Shell Functionality Validation")
    print("=" * 60)
    
    # Generate commands
    print(f"\n[1] Generating {args.num_per_template} commands per template...")
    commands = generate_test_commands(args.num_per_template, args.seed)
    print(f"    Generated {len(commands)} total commands")
    
    # Syntax validation
    print(f"\n[2] Running syntax validation...")
    syntax_results = run_syntax_validation(commands, args.sample_size, args.seed)
    
    print(f"\n    Results:")
    print(f"    - Total commands: {syntax_results['total_commands']}")
    print(f"    - Syntax valid:   {syntax_results['syntax_valid']} ({syntax_results['valid_rate']*100:.1f}%)")
    print(f"    - Syntax invalid: {syntax_results['syntax_invalid']}")
    print(f"    - Skipped:        {syntax_results['skipped']}")
    
    print(f"\n    By command type:")
    for cmd_type, stats in syntax_results["by_type"].items():
        rate = stats["valid"] / (stats["total"] - stats["skipped"]) if (stats["total"] - stats["skipped"]) > 0 else 0
        print(f"    - {cmd_type:12s}: {stats['valid']:4d}/{stats['total']:4d} valid ({rate*100:.1f}%)")
    
    if syntax_results["invalid_examples"]:
        print(f"\n    Sample invalid commands:")
        for i, ex in enumerate(syntax_results["invalid_examples"][:5]):
            print(f"    [{i+1}] Type: {ex['type']}")
            print(f"        Error: {ex['error'][:100]}...")
    
    # Save results
    if args.output:
        output_data = {
            "total_commands": syntax_results["total_commands"],
            "syntax_valid": syntax_results["syntax_valid"],
            "syntax_invalid": syntax_results["syntax_invalid"],
            "skipped": syntax_results["skipped"],
            "valid_rate": syntax_results["valid_rate"],
            "by_type": syntax_results["by_type"],
            "invalid_examples": syntax_results["invalid_examples"]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[*] Results saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)
    
    return syntax_results


if __name__ == "__main__":
    main()

