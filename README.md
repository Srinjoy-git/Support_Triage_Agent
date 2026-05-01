# Support Triage Agent

This is a terminal-based Python support triage agent. It reads support tickets
from CSV, retrieves relevant information from the local support corpus, decides
whether to reply or escalate, and writes predictions to an output CSV.

## Folder Structure

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── main.py          # Entry point
│   ├── classifier.py    # Request type and product area classification
│   ├── retrieval.py     # TF-IDF corpus retrieval with confidence score
│   ├── decision.py      # Reply vs escalation rules
│   ├── response.py      # Safe response generation
│   ├── utils.py         # Shared helpers for logging, config, and text cleanup
│   └── rules.json       # Configurable keywords and confidence threshold
├── data/                # Local support corpus used for retrieval
└── support_tickets/
    ├── support_tickets.csv
    ├── sample_support_tickets.csv
    └── output.csv
```

## How To Run In VS Code

Open this folder in VS Code, then run these commands from the integrated
terminal:

```bash
python3 -m pip install -r requirements.txt
python3 code/main.py
```

The program reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

## Workflow

1. `main.py` starts the terminal tool and prints progress logs:
   `Loading data...`, `Processing tickets...`, and `Saving output...`.
2. `retrieval.py` loads the markdown support corpus from `data/`, builds a
   TF-IDF index, and retrieves the most relevant document for each ticket.
3. `classifier.py` detects the ticket `request_type` and `product_area` using
   configurable keyword rules from `rules.json`.
4. `decision.py` decides whether the ticket should be `replied` or
   `escalated`. Sensitive issues and low-confidence retrieval matches are
   escalated.
5. `response.py` generates either a safe escalation message or a response
   grounded in the retrieved support document.
6. `main.py` saves the final results to `support_tickets/output.csv`.

## Output Columns

The output CSV contains:

```text
status, product_area, response, justification, request_type
```

Allowed `status` values:

```text
replied, escalated
```

Allowed `request_type` values:

```text
product_issue, feature_request, bug, invalid
```

