"""Job task: register the per-point model challenger, evaluate the promotion gates and move the champion
alias when allowed (pdm.entrypoints.promote)."""

from pdm.entrypoints import promote

if __name__ == "__main__":
    promote()
