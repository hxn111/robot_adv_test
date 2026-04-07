# robot_adv_test
adversarial test for cs763 final project

## Setting up
Installing dependencies:
```
pip install -r requirements.txt
```

Environment varaibles: create a '.env' file with
```
LOG_DIR="\\robot_adv_test\logs"
OPENAI_API_KEY="..."
```

To test:
```
python test_llm.py --group-size [number/unknown]
```

## Experiment Conditions + Strategy Gating
Condition definitions are in `strategy_conditions.py`:
1. `1`: baseline + single robot
2. `2`: humor + deflect + single robot
3. `3`: empathetic + deflect + single robot
4. `4`: humor + deflect + group dynamics
5. `5`: empathetic + deflect + group dynamics
6. `6`: humor + deflect + two-robot interactions
7. `7`: empathetic + deflect + two-robot interactions