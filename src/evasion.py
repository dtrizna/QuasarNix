import re
import random
from typing import List

def ip_to_decimal(ip):
    """
    Idea from: https://book.hacktricks.xyz/linux-hardening/bypass-bash-restrictions
    Implementation: ChatGPT.

    # Decimal IPs
    127.0.0.1 == 2130706433

    root@dmz:~# ping 2130706433
    PING 2130706433 (127.0.0.1) 56(84) bytes of data.
    64 bytes from 127.0.0.1: icmp_seq=1 ttl=64 time=0.030 ms
    64 bytes from 127.0.0.1: icmp_seq=2 ttl=64 time=0.036 ms
    ^C
    """
    parts = [int(part) for part in ip.split(".")]
    if len(parts) != 4:
        raise ValueError("The input does not seem to be a valid IPv4 address")    
    decimal_value = (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    return decimal_value


def replace_ip_with_decimal(command_adv):
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ip_match = re.search(ip_pattern, command_adv)
    if ip_match:
        ip_address = ip_match.group()
        try:
            decimal_ip = ip_to_decimal(ip_address)
            command_adv = command_adv.replace(ip_address, str(decimal_ip))
        except ValueError:
            print("A potential IP was found, but it's not valid. No replacement was made.")
    return command_adv


def attack_evasive_tricks(
        command: str,
        baseline: List[str], # keep for backward compatibility with attack_template_prepend
        attack_parameter: float = 0.7 # threshold how often to change, here: 70% of the time
) -> str:
    command_adv = command

    # replaces
    python_renames = [
            "cp /usr/bin/python /tmp/test; /tmp/test ",
            "cp /usr/bin/python /tmp/python; /tmp/python ",
        ]
    replace_maps = {
        # bash tricks
        "sh -i": [
            "sh -li",
            "sh -i -l",
            "sh -a -i",
            "sh -avi"
        ],
        # ";exec": [";id;exec", ";find /home 2>/dev/null;exec", ""], # not in templates
        ";cat": [";id;cat", ";readlink;cat", ";whoami;cat", ";find /home 2>/dev/null;cat"],

        # nc tricks
        "nc -e": ["nc -ne", "nc -v -e", "nc -env"],
        "nc ": ["ncat ", "nc.traditional "],
        
        # perl swaps
        "use Socket;": "use warnings; use Socket;",
        "perl -e": ["perl -S -e", "perl -t -e"],
        "perl ": [
            "cp /usr/bin/perl /tmp/test; /tmp/test ",
            "cp /usr/bin/perl /tmp/perl; /tmp/perl ",
        ],

        # php swaps
        "php -r": "php -e -r",
        "php ": "cp /usr/bin/php /tmp/test; /tmp/test ",

        # ruby swaps
        "ruby -rsocket": [
            "ruby -ruri -rsocket",
            "ruby -ryaml -rsocket"
        ],
        "-rsocket -e": "-rsocket -a -e",
        "-e'spawn": """-e'puts"test".inspect;spawn""",
        "ruby ": [
            "cp /usr/bin/ruby /tmp/test; /tmp/test ",
            "cp /usr/bin/ruby /tmp/ruby; /tmp/ruby "
        ],

        # python swaps
        "python -c": "python -b -c",
        "python3 -c": "python3 -b -c",
        "python ": "python2.7 ",
        "python ": python_renames,
        "python3 ": python_renames,
        "python2.7 ": python_renames,
        "import os": "import sys,os",
        "import socket": "import sys,socket",
        "os.system": "import os as bs;bs.system"
    }
    for replace_src, replace_dst in replace_maps.items():
        if replace_src in command_adv:
            chance = random.random()
            if chance <= attack_parameter:
                if isinstance(replace_dst, str):
                    command_adv = command_adv.replace(replace_src, replace_dst)
                elif isinstance(replace_dst, list):
                    command_adv = command_adv.replace(replace_src, random.choice(replace_dst))
    
    # ip manipulation
    chance = random.random()
    if chance <= attack_parameter:
        command_adv = replace_ip_with_decimal(command_adv)

    return command_adv
