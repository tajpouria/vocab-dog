test:
	pytest test_main.py -v

typecheck:
	mypy main.py

run:
	python main.py
