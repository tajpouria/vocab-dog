test:
	pytest

typecheck:
	mypy main.py

run:
	python main.py

deploy:
	ansible-playbook -i ansible/inventory/production ansible/deploy.yml

logs:
	ansible-playbook -i ansible/inventory/production ansible/logs.yml
