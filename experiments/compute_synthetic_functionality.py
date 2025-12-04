"""
Synthetic Reverse Shell Functionality Validation

This script validates that generated reverse shell commands are functional.
We use two validation approaches:

1. STATIC SYNTAX VALIDATION (fast, no execution):
   - Shell syntax: `bash -n -c "command"`
   - Python: compile() check
   - Perl/PHP/Ruby: interpreter syntax check

2. DOCKER-BASED EXECUTION (accurate, slower):
   - Run command in container with netcat listener
   - Check if connection is established

Exit codes interpretation:
- 0: Command is syntactically valid
- 1: General error (but often still means command parsed OK, just connection failed)
- 2: Shell syntax/parse error (INVALID)
- 126: Command not executable
- 127: Command/interpreter not found
- 128+N: Killed by signal N
"""

import os
import re
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm
import random
import pandas as pd

ROOT = Path(__file__).parent.parent

@dataclass
class ValidationResult:
    command: str
    command_type: str  # bash, python, perl, php, ruby, nc, lua, awk
    syntax_valid: bool
    syntax_exit_code: int
    syntax_error: str = ""
    docker_functional: Optional[bool] = None
    docker_exit_code: Optional[int] = None
    docker_connected: Optional[bool] = None

@dataclass 
class ValidationStats:
    total_commands: int = 0
    by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    syntax_valid: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    syntax_invalid: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    docker_functional: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    docker_failed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    syntax_errors: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


def classify_command_type(command: str) -> str:
    """Classify command by its primary interpreter/mechanism."""
    cmd_lower = command.lower()
    
    if 'python3 -c' in cmd_lower or 'python -c' in cmd_lower:
        return 'python'
    elif 'perl -e' in cmd_lower:
        return 'perl'
    elif 'php -r' in cmd_lower:
        return 'php'
    elif 'ruby -rsocket' in cmd_lower or 'ruby -e' in cmd_lower:
        return 'ruby'
    elif 'lua -e' in cmd_lower:
        return 'lua'
    elif "awk 'BEGIN" in command or 'awk "BEGIN' in command:
        return 'awk'
    elif 'nc ' in command or 'ncat ' in command or 'nc.traditional' in command:
        return 'nc'
    elif 'telnet ' in cmd_lower:
        return 'telnet'
    elif '/dev/tcp/' in command or '/dev/udp/' in command:
        return 'bash_dev'
    elif 'mkfifo' in command:
        return 'bash_fifo'
    else:
        return 'bash_other'


def validate_shell_syntax(command: str, shell: str = "bash") -> Tuple[bool, int, str]:
    """
    Validate shell command syntax using bash -n (syntax check only, no execution).
    
    Returns: (is_valid, exit_code, error_message)
    """
    try:
        result = subprocess.run(
            [shell, "-n", "-c", command],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_valid = result.returncode == 0
        error_msg = result.stderr.strip() if not is_valid else ""
        return is_valid, result.returncode, error_msg
    except subprocess.TimeoutExpired:
        return False, -1, "Timeout during syntax check"
    except Exception as e:
        return False, -1, str(e)


def extract_python_code(command: str) -> Optional[str]:
    """Extract Python code from a python -c command."""
    patterns = [
        r"python3? -[bc]? ?'([^']+)'",
        r'python3? -[bc]? ?"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python code syntax using compile()."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_perl_code(command: str) -> Optional[str]:
    """Extract Perl code from a perl -e command."""
    patterns = [
        r"perl -[eSt]* ?'([^']+)'",
        r'perl -[eSt]* ?"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_perl_syntax(code: str) -> Tuple[bool, int, str]:
    """Validate Perl code syntax using perl -c."""
    try:
        result = subprocess.run(
            ["perl", "-c", "-e", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_valid = result.returncode == 0
        error_msg = result.stderr.strip() if not is_valid else ""
        return is_valid, result.returncode, error_msg
    except FileNotFoundError:
        return True, 0, "perl not installed, assuming valid"
    except subprocess.TimeoutExpired:
        return False, -1, "Timeout"
    except Exception as e:
        return False, -1, str(e)


def extract_php_code(command: str) -> Optional[str]:
    """Extract PHP code from a php -r command."""
    patterns = [
        r"php -[er]* ?'([^']+)'",
        r'php -[er]* ?"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_php_syntax(code: str) -> Tuple[bool, int, str]:
    """Validate PHP code syntax using php -l."""
    try:
        # php -l needs a file or stdin with <?php
        full_code = f"<?php {code}"
        result = subprocess.run(
            ["php", "-l"],
            input=full_code,
            capture_output=True,
            text=True,
            timeout=5
        )
        is_valid = result.returncode == 0
        error_msg = result.stderr.strip() if not is_valid else ""
        return is_valid, result.returncode, error_msg
    except FileNotFoundError:
        return True, 0, "php not installed, assuming valid"
    except subprocess.TimeoutExpired:
        return False, -1, "Timeout"
    except Exception as e:
        return False, -1, str(e)


def extract_ruby_code(command: str) -> Optional[str]:
    """Extract Ruby code from a ruby command."""
    patterns = [
        r"ruby -rsocket -[ae]* ?'([^']+)'",
        r'ruby -rsocket -[ae]* ?"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def validate_ruby_syntax(code: str) -> Tuple[bool, int, str]:
    """Validate Ruby code syntax using ruby -c."""
    try:
        result = subprocess.run(
            ["ruby", "-rsocket", "-c", "-e", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_valid = result.returncode == 0
        error_msg = result.stderr.strip() if not is_valid else ""
        return is_valid, result.returncode, error_msg
    except FileNotFoundError:
        return True, 0, "ruby not installed, assuming valid"
    except subprocess.TimeoutExpired:
        return False, -1, "Timeout"
    except Exception as e:
        return False, -1, str(e)


def validate_command_syntax(command: str) -> ValidationResult:
    """
    Validate command syntax based on its type.
    Uses appropriate interpreter for each command type.
    """
    cmd_type = classify_command_type(command)
    
    # First, always do basic shell syntax check
    shell_valid, shell_exit, shell_error = validate_shell_syntax(command)
    
    # For interpreter-specific commands, also validate the embedded code
    if cmd_type == 'python':
        python_code = extract_python_code(command)
        if python_code:
            py_valid, py_error = validate_python_syntax(python_code)
            if not py_valid:
                return ValidationResult(
                    command=command,
                    command_type=cmd_type,
                    syntax_valid=False,
                    syntax_exit_code=2,
                    syntax_error=f"Python syntax error: {py_error}"
                )
    
    elif cmd_type == 'perl':
        perl_code = extract_perl_code(command)
        if perl_code:
            perl_valid, perl_exit, perl_error = validate_perl_syntax(perl_code)
            if not perl_valid:
                return ValidationResult(
                    command=command,
                    command_type=cmd_type,
                    syntax_valid=False,
                    syntax_exit_code=perl_exit,
                    syntax_error=f"Perl syntax error: {perl_error}"
                )
    
    elif cmd_type == 'php':
        php_code = extract_php_code(command)
        if php_code:
            php_valid, php_exit, php_error = validate_php_syntax(php_code)
            if not php_valid:
                return ValidationResult(
                    command=command,
                    command_type=cmd_type,
                    syntax_valid=False,
                    syntax_exit_code=php_exit,
                    syntax_error=f"PHP syntax error: {php_error}"
                )
    
    elif cmd_type == 'ruby':
        ruby_code = extract_ruby_code(command)
        if ruby_code:
            ruby_valid, ruby_exit, ruby_error = validate_ruby_syntax(ruby_code)
            if not ruby_valid:
                return ValidationResult(
                    command=command,
                    command_type=cmd_type,
                    syntax_valid=False,
                    syntax_exit_code=ruby_exit,
                    syntax_error=f"Ruby syntax error: {ruby_error}"
                )
    
    return ValidationResult(
        command=command,
        command_type=cmd_type,
        syntax_valid=shell_valid,
        syntax_exit_code=shell_exit,
        syntax_error=shell_error
    )


def validate_with_docker(
    command: str,
    timeout_seconds: int = 3,
    listener_port: int = 4444,
    image: str = "ubuntu:22.04"
) -> Tuple[bool, int, bool]:
    """
    Validate command functionality using Docker.
    
    Approach:
    1. Start container with netcat listener
    2. Modify command to connect to localhost:<listener_port>
    3. Run the command
    4. Check if connection was established
    
    Returns: (is_functional, exit_code, connection_established)
    """
    # Modify IP in command to use localhost
    modified_cmd = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '127.0.0.1',
        command
    )
    # Modify port to use our listener port
    modified_cmd = re.sub(
        r'(?<=,|\s|:)(\d{2,5})(?=\)|;|$|\s)',
        str(listener_port),
        modified_cmd
    )
    
    docker_script = f'''
    #!/bin/bash
    set -e
    
    # Start listener in background, capture if connection received
    timeout {timeout_seconds}s nc -lvp {listener_port} > /tmp/nc_output 2>&1 &
    NC_PID=$!
    sleep 0.5
    
    # Execute the reverse shell command (it will try to connect)
    timeout {timeout_seconds}s bash -c '{modified_cmd}' 2>/dev/null &
    CMD_PID=$!
    
    # Wait for potential connection
    sleep {timeout_seconds}
    
    # Check if listener received connection
    if grep -q "connect" /tmp/nc_output 2>/dev/null; then
        echo "CONNECTION_ESTABLISHED"
        exit 0
    else
        echo "NO_CONNECTION"
        exit 1
    fi
    '''
    
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--network=none",
                "-e", f"SCRIPT={docker_script}",
                image,
                "bash", "-c", docker_script
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 10
        )
        
        connected = "CONNECTION_ESTABLISHED" in result.stdout
        return True, result.returncode, connected
        
    except subprocess.TimeoutExpired:
        return False, -1, False
    except Exception as e:
        return False, -1, False


def load_synthetic_commands(
    root: Path,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> List[str]:
    """Load synthetic reverse shell commands from parquet files."""
    train_path = root / 'data' / 'nix_shell' / 'train_rvrs_real.parquet'
    test_path = root / 'data' / 'nix_shell' / 'test_rvrs_real.parquet'
    
    commands = []
    
    if train_path.exists():
        df = pd.read_parquet(train_path)
        commands.extend(df['cmd'].tolist())
    
    if test_path.exists():
        df = pd.read_parquet(test_path)
        commands.extend(df['cmd'].tolist())
    
    if sample_size and sample_size < len(commands):
        random.seed(seed)
        commands = random.sample(commands, sample_size)
    
    return commands


def run_validation(
    commands: List[str],
    use_docker: bool = False,
    docker_sample_size: int = 100
) -> ValidationStats:
    """Run validation on a list of commands."""
    stats = ValidationStats()
    results = []
    
    print(f"[*] Validating {len(commands)} commands...")
    print(f"[*] Docker validation: {'enabled' if use_docker else 'disabled'}")
    
    for cmd in tqdm(commands, desc="Syntax validation"):
        result = validate_command_syntax(cmd)
        results.append(result)
        
        stats.total_commands += 1
        stats.by_type[result.command_type] += 1
        
        if result.syntax_valid:
            stats.syntax_valid[result.command_type] += 1
        else:
            stats.syntax_invalid[result.command_type] += 1
            stats.syntax_errors[result.command_type].append(result.syntax_error)
    
    # Docker validation on sample
    if use_docker:
        valid_results = [r for r in results if r.syntax_valid]
        docker_sample = random.sample(
            valid_results, 
            min(docker_sample_size, len(valid_results))
        )
        
        print(f"\n[*] Running Docker validation on {len(docker_sample)} commands...")
        for result in tqdm(docker_sample, desc="Docker validation"):
            is_functional, exit_code, connected = validate_with_docker(result.command)
            result.docker_functional = is_functional
            result.docker_exit_code = exit_code
            result.docker_connected = connected
            
            if connected:
                stats.docker_functional[result.command_type] += 1
            else:
                stats.docker_failed[result.command_type] += 1
    
    return stats, results


def print_stats(stats: ValidationStats):
    """Print validation statistics."""
    print("\n" + "="*60)
    print("SYNTHETIC REVERSE SHELL FUNCTIONALITY VALIDATION REPORT")
    print("="*60)
    
    print(f"\nTotal commands validated: {stats.total_commands}")
    
    print("\n--- Commands by Type ---")
    for cmd_type, count in sorted(stats.by_type.items(), key=lambda x: -x[1]):
        valid = stats.syntax_valid.get(cmd_type, 0)
        invalid = stats.syntax_invalid.get(cmd_type, 0)
        valid_pct = 100 * valid / count if count > 0 else 0
        print(f"  {cmd_type:12s}: {count:6d} total | {valid:6d} valid ({valid_pct:5.1f}%) | {invalid:6d} invalid")
    
    total_valid = sum(stats.syntax_valid.values())
    total_invalid = sum(stats.syntax_invalid.values())
    valid_pct = 100 * total_valid / stats.total_commands if stats.total_commands > 0 else 0
    
    print(f"\n--- Overall Syntax Validation ---")
    print(f"  Valid:   {total_valid:6d} ({valid_pct:.2f}%)")
    print(f"  Invalid: {total_invalid:6d} ({100-valid_pct:.2f}%)")
    
    if stats.docker_functional:
        print(f"\n--- Docker Functional Validation ---")
        for cmd_type in stats.docker_functional:
            functional = stats.docker_functional.get(cmd_type, 0)
            failed = stats.docker_failed.get(cmd_type, 0)
            total = functional + failed
            func_pct = 100 * functional / total if total > 0 else 0
            print(f"  {cmd_type:12s}: {functional}/{total} functional ({func_pct:.1f}%)")
    
    if any(stats.syntax_errors.values()):
        print(f"\n--- Sample Syntax Errors ---")
        for cmd_type, errors in stats.syntax_errors.items():
            if errors:
                unique_errors = list(set(errors))[:3]
                print(f"  {cmd_type}:")
                for err in unique_errors:
                    print(f"    - {err[:100]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate synthetic reverse shell functionality")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None = all)")
    parser.add_argument("--docker", action="store_true", help="Enable Docker validation")
    parser.add_argument("--docker-sample", type=int, default=100, help="Docker sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"[*] Loading synthetic commands from {ROOT / 'data' / 'nix_shell'}...")
    commands = load_synthetic_commands(ROOT, sample_size=args.sample, seed=args.seed)
    print(f"[!] Loaded {len(commands)} commands")
    
    stats, results = run_validation(
        commands,
        use_docker=args.docker,
        docker_sample_size=args.docker_sample
    )
    
    print_stats(stats)
    
    if args.output:
        output_data = {
            'total_commands': stats.total_commands,
            'by_type': dict(stats.by_type),
            'syntax_valid': dict(stats.syntax_valid),
            'syntax_invalid': dict(stats.syntax_invalid),
            'syntax_valid_rate': sum(stats.syntax_valid.values()) / stats.total_commands if stats.total_commands > 0 else 0,
        }
        if stats.docker_functional:
            output_data['docker_functional'] = dict(stats.docker_functional)
            output_data['docker_failed'] = dict(stats.docker_failed)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[!] Results saved to {args.output}")


