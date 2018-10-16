#!/bin/bash

python3 multiarmed_bandits.py --mode greedy --epsilon 0.015625
python3 multiarmed_bandits.py --mode greedy --epsilon 0.03125
python3 multiarmed_bandits.py --mode greedy --epsilon 0.0625
python3 multiarmed_bandits.py --mode greedy --epsilon 0.125
python3 multiarmed_bandits.py --mode greedy --epsilon 0.25


python3 multiarmed_bandits.py --mode greedy --alpha 0.15 --epsilon 0.015625
python3 multiarmed_bandits.py --mode greedy --alpha 0.15 --epsilon 0.03125
python3 multiarmed_bandits.py --mode greedy --alpha 0.15 --epsilon 0.0625
python3 multiarmed_bandits.py --mode greedy --alpha 0.15 --epsilon 0.125
python3 multiarmed_bandits.py --mode greedy --alpha 0.15 --epsilon 0.25

python3 multiarmed_bandits.py --mode greedy --initial 1.0 --alpha 0.15 --epsilon 0.015625
python3 multiarmed_bandits.py --mode greedy --initial 1.0 --alpha 0.15 --epsilon 0.03125
python3 multiarmed_bandits.py --mode greedy --initial 1.0 --alpha 0.15 --epsilon 0.0625
python3 multiarmed_bandits.py --mode greedy --initial 1.0 --alpha 0.15 --epsilon 0.125
python3 multiarmed_bandits.py --mode greedy --initial 1.0 --alpha 0.15 --epsilon 0.25

python3 multiarmed_bandits.py --mode ucb --c 0.25
python3 multiarmed_bandits.py --mode ucb --c 0.5
python3 multiarmed_bandits.py --mode ucb --c 1
python3 multiarmed_bandits.py --mode ucb --c 2
python3 multiarmed_bandits.py --mode ucb --c 4

python3 multiarmed_bandits.py --mode gradient --alpha 0.0625
python3 multiarmed_bandits.py --mode gradient --alpha 0.125
python3 multiarmed_bandits.py --mode gradient --alpha 0.25
python3 multiarmed_bandits.py --mode gradient --alpha 0.5
