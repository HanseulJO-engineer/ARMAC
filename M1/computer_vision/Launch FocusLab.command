#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print "Create the .venv environment using the instructions in README.md first."
  read "reply?Press Enter to close."
  exit 1
fi
exec .venv/bin/python run_app.py
