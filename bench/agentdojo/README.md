# runtape on AgentDojo

[AgentDojo](https://github.com/ethz-spylab/agentdojo) is a public benchmark of tool-using agents
under prompt injection: realistic tasks in banking, Slack, travel and workspace environments, with
attacker text placed in the data the agent reads. Unlike the cases in `bench/`, nothing here was
written for runtape.

For each (user task, injection task) pair, `run.py`:

1. runs AgentDojo's own agent pipeline on the model, recording every model call with runtape;
2. if AgentDojo's security check says the attack succeeded, finds the model reply that made the
   attacker's call (identified by a value only the injection asks for, such as the attacker's
   account number);
3. runs `runtape why` on that decision without telling it where the injection is;
4. scores whether the headline cause lies inside the injected text, or is the whole tool result
   that holds it.

```
pip install agentdojo
python bench/agentdojo/run.py --openai sarvam-105b --base-url https://api.sarvam.ai/v1 --suite banking,slack --pairs 30
python bench/agentdojo/report.py bench/results/agentdojo-sarvam-105b.jsonl
```

The attack is AgentDojo's `important_instructions` without the model's name. A decision the model
no longer makes most of the time when `why` reruns it is reported separately, as in the main
benchmark. Traces are written to `bench/agentdojo/traces`, reruns are cached in
`bench/agentdojo/.cache`, and an interrupted run resumes where it stopped.

`tests/test_agentdojo.py` runs one pair end to end against a local stand-in server.
