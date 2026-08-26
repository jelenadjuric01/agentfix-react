"""agentgraph — a coding agent small enough to read end to end, built on LangGraph.

An agent here is a bounded graph that asks a model what to do, does it, and verifies the
result by running the tests rather than by believing the model.

This is the **ReAct** edition: the model reasons before it acts. Three siblings exist, and the
interesting parts of each are the places where it differs from the one before it:

  1. no framework, no reasoning   a hand-written `for` loop
  2. LangGraph, no reasoning      the same agent as a graph, driven by Mellum2 *Instruct*
  3. LangGraph + reasoning        this one, driven by Mellum2 *Thinking*

The step from 2 to 3 is one flag on the client (`reasoning=True`) and then the consequences of
it, which are not one flag. Suggested reading order:

  1. tools/base.py       what a tool is, the limits on what it may return, and the artifact
                         channel it reports through
  2. tasks/loader.py     what a task is; the copy-to-tempdir context manager
  3. tools/fs.py         list_files, read_file, write_file
  4. tools/tests_tool.py run_tests — the agent's only oracle
  5. llm/client.py       the one flag, and what it costs
  6. agent/state.py      what the graph carries, and the reducers that combine it
  7. agent/graph.py      the agent. If you read one file, read this one.
  8. runner.py           how the pieces above are wired together

Then, as needed: agent/trace.py (observability, and the one file the new model actively broke),
llm/fake.py (the scripted model), sandbox/ (how tests are executed), eval/ (measurement),
doctor.py and cli.py (entry points).

agent/prebuilt.py is the argument rather than the implementation: the same agent built from
`langchain.agents.create_agent` and its middleware. Optional dependency, not on the path
`agentgraph solve` takes.
"""

__version__ = "0.1.0"
