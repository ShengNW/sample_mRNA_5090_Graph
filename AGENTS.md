- To execute commands on the remote training server, use:

  - `./r.sh "<COMMAND>"`

  Examples:
  - `./r.sh "make mfe"`
  - `./r.sh "nohup make mfe > logs/mfe.log 2>&1 &"`

- Assume that the working directory on the remote side is:
  `/root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2`.

- You may call `./r.sh` instead of running heavy commands locally.
