## Setup

```bash
python3 -m venv venv;
source venv/bin/activate;
pip install -r requirements.txt;
``` 

## Structure
- [Results Folder](results/)
- [SW Graphs Folder](sw_graphs/)
- [Random Graph Generator](random_graph_generator.py)
- **Algorithms**:
  - [Random Max Weighted Matching Algorithm](random_max_weighted_matching.py) - covered in report
  - [Probabilistic Greedy Algorithm](probabilistic_greedy_search.py) - covered in report
  - [Random Max Weighted Matching With Heuristics](random_max_weighted_matching_with_heuristic.py) - not covered in report
- **Mains**:
  - [main.py](main.py) - cli for running the algorithms with the random generated graphs. Run [random_graph_generator.py](random_graph_generator.py) first to generate the graphs
  - [main_sw_graphs.py](main_sw_graphs.py) - cli for running the algorithms with the sw graphs
- **Report**: [AA_Assignment2_108287_report.pdf](AA_Assignment2_108287_report.pdf)